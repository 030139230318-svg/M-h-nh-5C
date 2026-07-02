# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import plotly.express as px
import plotly.figure_factory as ff
import io

st.set_page_config(
    page_title="Dự Báo Rủi Ro - Logistic Regression",
    page_icon="📊",
    layout="wide"
)

# ====================== HÀM CACHE DỮ LIỆU ======================
@st.cache_data
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader(
        "Tải file dữ liệu CSV", 
        type=["csv"],
        help="File phải có các cột: TC1..TS4, PD (target)"
    )
    
    st.divider()
    st.subheader("Tham số mô hình")
    
    test_size = st.slider("Tỷ lệ tập test", 0.1, 0.4, 0.2, 0.05, help="Tỷ lệ dữ liệu dùng để kiểm tra")
    random_state = st.number_input("Random State", value=42, min_value=0, help="Giá trị seed để tái tạo kết quả")
    
    st.divider()
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ====================== HEADER ======================
st.title("📊 Hệ Thống Dự Báo Rủi Ro Khách Hàng")
st.caption("Mô hình Logistic Regression dự báo PD (rủi ro) dựa trên các chỉ số đánh giá (TC, NL, DK, V, TS)")

if uploaded_file is None:
    st.info("👆 Vui lòng tải file CSV dữ liệu lên từ sidebar để bắt đầu.")
    st.stop()

df = load_data(uploaded_file.getvalue())
st.caption(f"📁 Đang sử dụng file: **{uploaded_file.name}** - {df.shape[0]} dòng, {df.shape[1]} cột")

# ====================== CÁC BIẾN ======================
feature_cols = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4',
                'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5',
                'V6', 'TS1', 'TS2', 'TS3', 'TS4']
target_col = 'PD'

if target_col not in df.columns:
    st.error("File dữ liệu thiếu cột PD (biến mục tiêu).")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# ====================== TRAIN ======================
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.accuracy = None

if train_button:
    with st.spinner("Đang huấn luyện mô hình..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
      model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Lưu vào session_state
        st.session_state.model = model
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.accuracy = acc
        
        st.success(f"✅ Huấn luyện hoàn tất! Độ chính xác trên tập test: **{acc:.4f}**")

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Tổng quan dữ liệu", "📈 Trực quan hóa", "📊 Kết quả huấn luyện", "🔮 Sử dụng mô hình"])

# TAB 1: TỔNG QUAN
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số quan sát", len(df))
    with col2:
        st.metric("Số đặc trưng", len(feature_cols))
    with col3:
        st.metric("Tỷ lệ rủi ro (PD=1)", f"{y.mean():.1%}")
    
    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Thống kê mô tả các đặc trưng")
    st.dataframe(X.describe(), use_container_width=True)

# TAB 2: TRỰC QUAN
with tab2:
    st.subheader("Phân bố biến mục tiêu PD")
    fig_target = px.histogram(df, x='PD', color='PD', barmode='group', title="Phân bố Rủi ro")
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.subheader("Chọn biến để trực quan")
    selected_vars = st.multiselect("Chọn biến", feature_cols, default=feature_cols[:4])
    
    cols = st.columns(2)
    for i, var in enumerate(selected_vars[:4]):
        with cols[i % 2]:
            fig = px.histogram(df, x=var, color='PD', title=f"Phân bố {var} theo PD")
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: KẾT QUẢ
with tab3:
    if st.session_state.model is None:
        st.info("👈 Vui lòng huấn luyện mô hình từ sidebar trước.")
    else:
        st.success(f"**Độ chính xác tổng thể:** {st.session_state.accuracy:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            fig_cm = ff.create_annotated_heatmap(
                cm, x=['Dự báo 0', 'Dự báo 1'], y=['Thực 0', 'Thực 1'],
                colorscale='Blues', showscale=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("Báo cáo phân loại")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
          st.dataframe(report_df.round(4), use_container_width=True)

# TAB 4: SỬ DỤNG
with tab4:
    if st.session_state.model is None:
        st.info("👈 Vui lòng huấn luyện mô hình từ sidebar trước.")
    else:
        mode = st.radio("Chế độ dự báo", ["Nhập thủ công", "Tải file CSV"])
        
        if mode == "Nhập thủ công":
            st.subheader("Nhập giá trị các chỉ số")
            input_data = {}
            cols_input = st.columns(4)
            
            for i, col in enumerate(feature_cols):
                with cols_input[i % 4]:
                    # Giá trị mặc định là trung vị
                    default_val = int(X[col].median())
                    input_data[col] = st.number_input(
                        col, min_value=1, max_value=5, value=default_val, step=1
                    )
            
            if st.button("🔮 Dự báo", type="primary"):
                input_df = pd.DataFrame([input_data])
                pred = st.session_state.model.predict(input_df)[0]
                proba = st.session_state.model.predict_proba(input_df)[0]
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Kết quả dự báo", "🚨 Có rủi ro" if pred == 1 else "✅ Không rủi ro")
                with col_p2:
                    st.metric("Xác suất rủi ro", f"{proba[1]:.1%}")
        
        else:  # Tải file
            pred_file = st.file_uploader("Tải file CSV cần dự báo (cần đủ các cột đặc trưng)", type=["csv"])
            if pred_file:
                try:
                    pred_df = pd.read_csv(io.BytesIO(pred_file.getvalue()))
                    missing_cols = [c for c in feature_cols if c not in pred_df.columns]
                    if missing_cols:
                        st.error(f"Thiếu cột: {missing_cols}")
                    else:
                        X_pred = pred_df[feature_cols]
                        preds = st.session_state.model.predict(X_pred)
                        probas = st.session_state.model.predict_proba(X_pred)[:, 1]
                        
                        result_df = pred_df.copy()
                        result_df['Dự báo_PD'] = preds
                        result_df['Xác suất_rủi_ro'] = probas
                        
                        st.subheader("Kết quả dự báo")
                        st.dataframe(result_df, use_container_width=True)
                        
                        csv = result_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            "📥 Tải kết quả CSV",
                            csv,
                            "ket_qua_du_bao.csv",
                            "text/csv",
                            key='download-csv'
                        )
                except Exception as e:
st.error(f"Lỗi khi xử lý file: {str(e)}")

st.caption("Mô hình Logistic Regression | Dữ liệu huấn luyện từ khảo sát đánh giá")
