# frontend/streamlit/pages/pipeline_dashboard.py
"""
Dashboard thống kê tình hình xử lý theo Data Lake Pipeline.
Hiển thị số lượng theo từng zone và từ catalog.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import DATA_ROOT
from utils.sqlite_viewer import copy_db_to_temp
from state.session import require_login
from services.api_client import get_me, admin_list_users, admin_delete_user_messages

# Cache 60s để tránh đọc NAS/DB liên tục
CACHE_TTL = 60

ZONES = {
    "000_inbox": ("000_inbox", "Inbox", "Chờ ingest"),
    "100_raw": ("100_raw", "Raw", "Đã ingest"),
    "200_staging": ("200_staging", "Staging", "Đã staging"),
    "300_processed": ("300_processed", "Processed", "Đã xử lý"),
    "400_embeddings": ("400_embeddings", "Embeddings", "Đã embed"),
    "500_catalog": ("500_catalog", "Catalog", "Metadata"),
}


@st.cache_data(ttl=CACHE_TTL)
def _count_inbox_files() -> int:
    """Đếm số file trong 000_inbox (một cấp domain, mỗi domain đếm file)."""
    inbox = DATA_ROOT / "000_inbox"
    if not inbox.exists():
        return 0
    total = 0
    try:
        for d in inbox.iterdir():
            if d.name.startswith(".") or not d.is_dir():
                continue
            for _ in d.iterdir():
                total += 1
                if total > 50_000:  # giới hạn tránh treo
                    return total
    except (PermissionError, OSError):
        pass
    return total


def _read_catalog_count(table: str) -> int | None:
    """Đọc COUNT từ catalog — copy DB ra temp tránh Errno 35 (resource deadlock)."""
    if table not in ("raw_objects", "ingest_log"):
        return None
    db = DATA_ROOT / "500_catalog" / "catalog.sqlite"
    if not db.exists():
        return None
    temp_path = None
    try:
        temp_path = copy_db_to_temp(db)
        conn = sqlite3.connect(str(temp_path), timeout=5)
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return None
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@st.cache_data(ttl=CACHE_TTL)
def _count_raw_objects_catalog() -> int | None:
    """Đếm số bản ghi trong raw_objects (catalog)."""
    return _read_catalog_count("raw_objects")


@st.cache_data(ttl=CACHE_TTL)
def _count_ingest_log() -> int | None:
    """Đếm số bản ghi ingest_log."""
    return _read_catalog_count("ingest_log")


@st.cache_data(ttl=CACHE_TTL)
def _count_zone_dirs(zone_key: str) -> int:
    """
    Đếm số "item" trong zone: với 100_raw = số file (domain/hash.ext);
    với 200/300/400 = số thư mục con (domain/hash).
    """
    path = DATA_ROOT / ZONES[zone_key][0]
    if not path.exists():
        return 0
    total = 0
    try:
        for domain in path.iterdir():
            if domain.name.startswith(".") or not domain.is_dir():
                continue
            for _ in domain.iterdir():
                total += 1
                if total > 100_000:
                    return total
    except (PermissionError, OSError):
        pass
    return total


@st.cache_data(ttl=CACHE_TTL)
def _count_raw_files() -> int:
    """Đếm file trong 100_raw (domain/hash.ext)."""
    return _count_zone_dirs("100_raw")


def _get_pipeline_stats() -> dict:
    """Tổng hợp số liệu cho dashboard."""
    inbox_count = _count_inbox_files()
    raw_catalog = _count_raw_objects_catalog()
    raw_files = _count_raw_files()
    staging_count = _count_zone_dirs("200_staging")
    processed_count = _count_zone_dirs("300_processed")
    embeddings_count = _count_zone_dirs("400_embeddings")
    ingest_log_count = _count_ingest_log()

    return {
        "000_inbox": {"count": inbox_count, "label": "File chờ ingest"},
        "100_raw": {
            "count": raw_files,
            "catalog": raw_catalog,
            "label": "File raw (catalog: " + (str(raw_catalog) if raw_catalog is not None else "—") + ")",
        },
        "200_staging": {"count": staging_count, "label": "Thư mục staging"},
        "300_processed": {"count": processed_count, "label": "Thư mục processed"},
        "400_embeddings": {"count": embeddings_count, "label": "Thư mục embeddings"},
        "500_catalog": {
            "raw_objects": raw_catalog,
            "ingest_log": ingest_log_count,
            "label": "Catalog DB",
        },
    }


def render():
    if not require_login():
        return

    st.header("📊 Dashboard")
    st.caption("Thống kê tình hình xử lý theo từng bước Data Lake Pipeline. Số liệu cache 60s.")

    if not DATA_ROOT.exists():
        st.warning(f"Data root chưa tồn tại: {DATA_ROOT}")
        return

    if st.button("🔄 Làm mới số liệu", help="Xóa cache và tải lại"):
        _count_inbox_files.clear()
        _count_raw_objects_catalog.clear()
        _count_ingest_log.clear()
        _count_zone_dirs.clear()
        _count_raw_files.clear()
        st.rerun()

    try:
        stats = _get_pipeline_stats()
    except Exception as e:
        st.error(f"Lỗi đọc thống kê: {e}")
        return

    # ---------- Thẻ số liệu theo zone ----------
    st.subheader("Số lượng theo zone")
    cols = st.columns(6)
    zone_order = ["000_inbox", "100_raw", "200_staging", "300_processed", "400_embeddings", "500_catalog"]
    for i, key in enumerate(zone_order):
        zkey, ztitle, _ = ZONES[key]
        with cols[i]:
            s = stats.get(key, {})
            if key == "500_catalog":
                raw_n = s.get("raw_objects")
                log_n = s.get("ingest_log")
                st.metric("Catalog", f"raw: {raw_n if raw_n is not None else '—'}", f"log: {log_n if log_n is not None else '—'}")
            else:
                count = s.get("count", 0)
                st.metric(ztitle, str(count), "")
    st.divider()

    # ---------- Biểu đồ: Pipeline theo bước ----------
    st.subheader("📈 Luồng pipeline (biểu đồ)")
    pipeline_labels = ["Inbox", "Raw", "Staging", "Processed", "Embeddings"]
    pipeline_counts = [
        stats["000_inbox"]["count"],
        stats["100_raw"]["count"],
        stats["200_staging"]["count"],
        stats["300_processed"]["count"],
        stats["400_embeddings"]["count"],
    ]
    df_pipeline = pd.DataFrame({"Bước": pipeline_labels, "Số lượng": pipeline_counts})
    ch1, ch2 = st.columns(2)
    with ch1:
        st.bar_chart(df_pipeline.set_index("Bước"), height=280)
    with ch2:
        st.area_chart(df_pipeline.set_index("Bước"), height=280)
    st.caption("Inbox → Raw → Staging → Processed → Embeddings → Qdrant")

    # ---------- Biểu đồ: So sánh zone (cột) ----------
    st.subheader("📊 So sánh số lượng theo zone")
    zone_titles = [ZONES[k][1] for k in zone_order]
    zone_counts = []
    for k in zone_order:
        s = stats.get(k, {})
        if k == "500_catalog":
            zone_counts.append(s.get("raw_objects") or 0)
        else:
            zone_counts.append(s.get("count", 0))
    df_zones = pd.DataFrame({"Zone": zone_titles, "Số lượng": zone_counts})
    st.bar_chart(df_zones.set_index("Zone"), height=300)
    st.divider()

    # ---------- Thống kê tin nhắn Q&A (Admin) ----------
    st.subheader("👤 Thống kê tin nhắn Q&A")
    st.caption("Số tin nhắn (câu hỏi Q&A) mỗi tài khoản. Chỉ admin có thể xóa toàn bộ tin nhắn của một user.")
    token = st.session_state.get("token")
    me = get_me(token) if token else None
    current_username = me.get("username") if me else None
    is_admin = current_username == "admin"
    try:
        users = admin_list_users(token) if token else []
    except Exception as exc:
        st.warning(f"Không tải được danh sách user: {exc}")
        users = []
    if not users:
        st.info("Chưa có user nào có tin nhắn trong hệ thống.")
    else:
        # Biểu đồ cột: tin nhắn theo user
        df_msgs = pd.DataFrame([
            {"User": u.get("username", ""), "Số tin nhắn": u.get("message_count", 0)}
            for u in users
        ])
        st.bar_chart(df_msgs.set_index("User"), height=260)
        st.markdown("**Chi tiết & thao tác**")
        for u in users:
            username = u.get("username", "")
            count = u.get("message_count", 0)
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                st.write("**" + username + "**")
            with c2:
                st.metric("Số tin nhắn", count)
            with c3:
                if is_admin:
                    if st.button("🗑️ Xoá toàn bộ tin nhắn", key=f"dashboard_del_{username}", type="secondary"):
                        try:
                            result = admin_delete_user_messages(username, token)
                            st.success(f"Đã xóa {result.get('deleted_count', 0)} tin nhắn của **{username}**.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi khi xóa: {e}")
                else:
                    st.caption("(Chỉ admin mới xóa được)")
            st.divider()

    st.caption("Dữ liệu đọc từ filesystem và 500_catalog/catalog.sqlite. Pipeline Runner dùng để chạy từng bước.")
