import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from services.process_ans import (
    find_answer_blocks,
    get_answers,
    process_ans_blocks,
    process_list_ans,
)
from services.process_img import warp_process

st.set_page_config(
    page_title="OMR - Chấm bài trắc nghiệm",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Hệ thống chấm bài trắc nghiệm OMR(Test CI/CD) 2")
st.write(
    "Tải ảnh phiếu trả lời và file JSON đáp án chuẩn "
    "để nhận diện, đối chiếu và tính điểm tự động."
)

# Khởi tạo session_state để lưu trữ kết quả
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "warped" not in st.session_state:
    st.session_state.warped = None


def normalize_answer(value):
    """Chuẩn hóa đáp án về A/B/C/D hoặc None."""
    if value is None:
        return None

    answer = str(value).strip().upper()

    if answer in ("", "NONE", "NULL", "-", "—"):
        return None

    if answer not in ("A", "B", "C", "D"):
        raise ValueError(f"Đáp án không hợp lệ: {value}")

    return answer


def load_answer_key(uploaded_file):
    """Đọc JSON dạng {'answers': {'1': 'A'}} hoặc dict trực tiếp."""
    try:
        data = json.load(uploaded_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("File không phải JSON hợp lệ.") from exc

    if isinstance(data, dict) and "answers" in data:
        data = data["answers"]

    if isinstance(data, list):
        raw_answers = {
            str(index + 1): value for index, value in enumerate(data)
        }
    elif isinstance(data, dict):
        raw_answers = data
    else:
        raise ValueError(
            "JSON phải là object đáp án hoặc danh sách đáp án."
        )

    answer_key = {}

    for question, answer in raw_answers.items():
        try:
            question_number = int(question)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Số câu không hợp lệ: {question}") from exc

        if not 1 <= question_number <= 120:
            raise ValueError(f"Câu {question_number} nằm ngoài phạm vi 1–120.")

        answer_key[question_number] = normalize_answer(answer)

    return answer_key


def grade_answers(results, answer_key):
    """So sánh đáp án nhận diện với đáp án chuẩn."""
    rows = []

    for question in range(1, 121):
        result = results.get(question, results.get(str(question), {}))
        if not isinstance(result, dict):
            result = {}

        detected = normalize_answer(result.get("answer"))
        correct = answer_key.get(question)

        if correct is None:
            status = "Chưa có đáp án chuẩn"
        elif detected is None:
            status = "Bỏ trống"
        elif detected == correct:
            status = "Đúng"
        else:
            status = "Sai"

        rows.append(
            {
                "Câu": question,
                "Đáp án chuẩn": correct or "—",
                "Đáp án nhận diện": detected or "—",
                "Kết quả": status,
                "Độ tin cậy": result.get("ratio"),
            }
        )

    df = pd.DataFrame(rows)

    graded = df[df["Đáp án chuẩn"] != "—"]
    correct_count = int((graded["Kết quả"] == "Đúng").sum())
    wrong_count = int((graded["Kết quả"] == "Sai").sum())
    blank_count = int((graded["Kết quả"] == "Bỏ trống").sum())
    total_graded = len(graded)

    score = correct_count / total_graded * 10 if total_graded > 0 else 0.0

    summary = {
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": total_graded,
        "score": score,
    }

    return df, summary


# --- Khu vực tải file ---
left, right = st.columns(2)

with left:
    image_file = st.file_uploader(
        "1. Tải ảnh phiếu trả lời",
        type=["png", "jpg", "jpeg"],
    )

with right:
    json_file = st.file_uploader(
        "2. Tải file JSON đáp án",
        type=["json"],
    )

weights_path = "weights/weighted.h5"

if image_file:
    # Đọc bytes và giải mã bằng OpenCV để hiển thị ổn định trên Docker (Linux)
    image_bytes = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Chuyển đổi từ BGR sang RGB để st.image hiển thị đúng màu trên mọi nền tảng
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(
            image_rgb,
            caption="Ảnh phiếu được tải lên",
            use_container_width=False,
        )
    else:
        st.error("Không thể đọc được file ảnh. Vui lòng thử định dạng khác (PNG, JPG).")

if st.button("🔍 Nhận diện và chấm điểm", type="primary"):
    if image_file is None or json_file is None:
        st.error("Vui lòng tải cả ảnh phiếu và file JSON đáp án.")
        st.stop()

    # 1. Đọc đáp án chuẩn
    try:
        answer_key = load_answer_key(json_file)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not answer_key or not any(
        answer is not None for answer in answer_key.values()
    ):
        st.error("File JSON chưa có đáp án chuẩn hợp lệ.")
        st.stop()

    # 2. Đọc ảnh từ dữ liệu upload
    image_bytes = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Không đọc được ảnh. Hãy thử một file ảnh khác.")
        st.stop()

    # 3. Chỉnh phối cảnh bằng thuật toán hiện có
    with st.spinner("Đang xử lý ảnh và nhận diện ô trả lời..."):
        try:
            warped = warp_process(image)

            if warped is None:
                st.error(
                    "Không tìm được biên phiếu. "
                    "Hãy sử dụng ảnh rõ, chụp đầy đủ phiếu và tránh nghiêng quá nhiều."
                )
                st.stop()

            # 4. Tìm các khối đáp án
            blocks = find_answer_blocks(warped)

            if blocks is None or len(blocks) == 0:
                st.error("Không tìm thấy khối đáp án trên phiếu.")
                st.stop()

            # 5. Tách dòng và từng ô lựa chọn
            lines = process_ans_blocks(blocks)
            choices = process_list_ans(lines)

            # Pipeline hiện tại yêu cầu 120 câu x 4 lựa chọn.
            if choices is None or len(choices) != 480:
                actual = 0 if choices is None else len(choices)
                st.error(
                    f"Thuật toán tách được {actual} ô, "
                    "nhưng cần đúng 480 ô (120 câu × 4 lựa chọn). "
                    "Hãy kiểm tra ảnh hoặc thuật toán tách ô."
                )
                st.stop()

            # 6. Chạy bộ phân loại hiện có
            results = get_answers(
                choices,
                model_path=weights_path,
            )

            if not results:
                st.error("Không nhận diện được đáp án từ phiếu.")
                st.stop()

            # 7. Chấm điểm và lưu vào session_state
            df, summary = grade_answers(results, answer_key)
            st.session_state.df = df
            st.session_state.summary = summary
            st.session_state.warped = warped

        except Exception as exc:
            st.error(f"Xử lý thất bại: {exc}")
            st.exception(exc)
            st.stop()

    st.success("Đã hoàn tất nhận diện và chấm điểm.")

# --- Hiển thị kết quả (độc lập với nút bấm để giữ trạng thái khi lọc) ---
if st.session_state.df is not None:
    df = st.session_state.df
    summary = st.session_state.summary
    warped = st.session_state.warped

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Điểm", f"{summary['score']:.2f}/10")
    m2.metric("Số câu đúng", summary["correct"])
    m3.metric("Số câu sai", summary["wrong"])
    m4.metric("Bỏ trống", summary["blank"])

    st.subheader("Bảng kết quả từng câu")

    filter_status = st.selectbox(
        "Lọc kết quả",
        [
            "Tất cả",
            "Đúng",
            "Sai",
            "Bỏ trống",
            "Chưa có đáp án chuẩn",
        ],
    )

    display_df = df
    if filter_status != "Tất cả":
        display_df = df[df["Kết quả"] == filter_status]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # --- Tải kết quả CSV ---
    csv_data = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải kết quả CSV",
        data=csv_data,
        file_name="ket_qua_cham_bai.csv",
        mime="text/csv",
    )

    with st.expander("Xem ảnh sau khi chỉnh phối cảnh"):
        if warped is not None:
            # Kiểm tra xem ảnh là ảnh màu (3 kênh) hay ảnh xám (1 kênh)
            if len(warped.shape) == 3 and warped.shape[2] == 3:
                warped_display = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            elif len(warped.shape) == 2:
                # Nếu là ảnh xám, chuyển đổi sang RGB để st.image hiển thị chuẩn
                warped_display = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
            else:
                warped_display = warped

            st.image(
                warped_display,
                caption="Ảnh đã qua warp_process()",
                use_container_width=True,
            )
        else:
            st.warning("Không có dữ liệu ảnh sau khi xử lý phối cảnh.")