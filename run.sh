#!/bin/bash
# ═══════════════════════════════════════════════════
#  DSS — Script Chạy Nhanh
#  Hệ thống Hỗ trợ Quyết định Mua Bán Cổ Phiếu VN30
# ═══════════════════════════════════════════════════

set -e

# ── Đường dẫn dự án ──
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# ── Màu sắc ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  📈 DSS — $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════${NC}"
    echo ""
}

# ── Kích hoạt venv nếu có ──
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -d "$HOME/.venv" ]; then
        source "$HOME/.venv/bin/activate"
    fi
}

# ═══════════════ COMMANDS ═══════════════

cmd_setup() {
    print_header "Setup Môi Trường"

    # Tạo venv nếu chưa có
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}📦 Tạo virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    echo -e "${YELLOW}📦 Cài đặt thư viện...${NC}"
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"

    # Cài thêm thư viện cần thiết nếu chưa có trong requirements.txt
    pip install ta scikit-learn xgboost joblib tabulate colorama

    echo -e "${YELLOW}📁 Tạo thư mục...${NC}"
    mkdir -p "$PROJECT_DIR/data/stocks"
    mkdir -p "$PROJECT_DIR/data/index"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/notebooks"

    echo ""
    echo -e "${GREEN}${BOLD}✅ Setup hoàn tất!${NC}"
    echo -e "   Tiếp theo chạy: ${CYAN}./run.sh fetch${NC} để tải dữ liệu VN30"
}

cmd_fetch() {
    print_header "Tải Dữ Liệu VN30"
    activate_venv

    echo -e "${YELLOW}📡 Đang quét rổ VN30 và tải OHLCV...${NC}"
    python3 "$PROJECT_DIR/src/data_fetcher.py"

    echo ""
    echo -e "${GREEN}${BOLD}✅ Tải dữ liệu hoàn tất!${NC}"
    echo -e "   Dữ liệu lưu tại: ${CYAN}$PROJECT_DIR/data/${NC}"
}

cmd_dss() {
    print_header "Chạy Khuyến Nghị Hôm Nay"
    activate_venv

    if [ ! -f "$PROJECT_DIR/main.py" ]; then
        echo -e "${RED}❌ Chưa có file main.py. Hãy tạo theo DSS_FULL_CODE_GUIDE.md${NC}"
        exit 1
    fi

    python3 "$PROJECT_DIR/main.py"
}

cmd_backtest() {
    print_header "Kiểm Chứng Lịch Sử (Backtest)"
    activate_venv

    if [ ! -f "$PROJECT_DIR/backtest_runner.py" ]; then
        echo -e "${RED}❌ Chưa có file backtest_runner.py. Hãy tạo theo DSS_FULL_CODE_GUIDE.md${NC}"
        exit 1
    fi

    python3 "$PROJECT_DIR/backtest_runner.py"
}

cmd_test_phase() {
    PHASE=$1
    activate_venv

    case $PHASE in
        1) print_header "Test Phase 1: Data Fetcher"
           python3 "$PROJECT_DIR/src/data_fetcher.py" ;;
        2) print_header "Test Phase 2: Data Cleaner"
           python3 "$PROJECT_DIR/src/data_cleaner.py" ;;
        3) print_header "Test Phase 3: Indicators"
           python3 "$PROJECT_DIR/src/indicators.py" ;;
        4) print_header "Test Phase 4: Features"
           python3 "$PROJECT_DIR/src/features.py" ;;
        5) print_header "Test Phase 5: ML Models"
           python3 "$PROJECT_DIR/src/ml_models.py" ;;
        *) echo -e "${RED}❌ Phase không hợp lệ. Chọn 1-5.${NC}" ;;
    esac
}

cmd_clean() {
    print_header "Dọn dẹp"

    echo -e "${YELLOW}🗑️  Xóa data cache...${NC}"
    rm -rf "$PROJECT_DIR/data/stocks/"*.csv
    rm -rf "$PROJECT_DIR/data/index/"*.csv

    echo -e "${YELLOW}🗑️  Xóa models đã train...${NC}"
    rm -rf "$PROJECT_DIR/models/"*.pkl

    echo -e "${YELLOW}🗑️  Xóa __pycache__...${NC}"
    find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    echo -e "${GREEN}✅ Đã dọn sạch!${NC}"
}

cmd_status() {
    print_header "Trạng Thái Dự Án"

    echo -e "${BOLD}📂 Dữ liệu:${NC}"
    if [ -d "$PROJECT_DIR/data/stocks" ]; then
        CSV_COUNT=$(ls "$PROJECT_DIR/data/stocks/"*.csv 2>/dev/null | wc -l | tr -d ' ')
        echo -e "   Stocks CSV: ${CYAN}${CSV_COUNT} file${NC}"
    else
        echo -e "   Stocks CSV: ${RED}chưa có${NC}"
    fi

    if [ -f "$PROJECT_DIR/data/index/VNINDEX.csv" ]; then
        INDEX_ROWS=$(wc -l < "$PROJECT_DIR/data/index/VNINDEX.csv" | tr -d ' ')
        echo -e "   VNINDEX:    ${CYAN}${INDEX_ROWS} dòng${NC}"
    else
        echo -e "   VNINDEX:    ${RED}chưa có${NC}"
    fi

    echo ""
    echo -e "${BOLD}🤖 Models:${NC}"
    if [ -d "$PROJECT_DIR/models" ]; then
        PKL_COUNT=$(ls "$PROJECT_DIR/models/"*.pkl 2>/dev/null | wc -l | tr -d ' ')
        echo -e "   Trained:    ${CYAN}${PKL_COUNT} file .pkl${NC}"
    else
        echo -e "   Trained:    ${RED}chưa có${NC}"
    fi

    echo ""
    echo -e "${BOLD}📝 Source files:${NC}"
    for f in data_fetcher data_cleaner indicators features ml_models scoring decision backtester; do
        if [ -f "$PROJECT_DIR/src/${f}.py" ]; then
            echo -e "   src/${f}.py   ${GREEN}✅${NC}"
        else
            echo -e "   src/${f}.py   ${RED}❌ chưa tạo${NC}"
        fi
    done

    for f in main.py backtest_runner.py; do
        if [ -f "$PROJECT_DIR/$f" ]; then
            echo -e "   ${f}            ${GREEN}✅${NC}"
        else
            echo -e "   ${f}            ${RED}❌ chưa tạo${NC}"
        fi
    done
}

cmd_help() {
    print_header "Hướng Dẫn Sử Dụng"

    echo -e "${BOLD}Cú pháp:${NC} ./run.sh <lệnh>"
    echo ""
    echo -e "${BOLD}Các lệnh:${NC}"
    echo -e "  ${CYAN}setup${NC}        Tạo venv, cài thư viện, tạo thư mục"
    echo -e "  ${CYAN}fetch${NC}        Tải dữ liệu VN30 mới nhất từ vnstock"
    echo -e "  ${CYAN}dss${NC}          Chạy khuyến nghị hôm nay (main.py)"
    echo -e "  ${CYAN}backtest${NC}     Chạy kiểm chứng lịch sử (backtest_runner.py)"
    echo -e "  ${CYAN}test <1-5>${NC}   Test từng Phase riêng lẻ"
    echo -e "  ${CYAN}status${NC}       Kiểm tra trạng thái dự án (data, models, files)"
    echo -e "  ${CYAN}clean${NC}        Xóa cache data, models, __pycache__"
    echo -e "  ${CYAN}help${NC}         Hiển thị hướng dẫn này"
    echo ""
    echo -e "${BOLD}Quy trình đề xuất:${NC}"
    echo -e "  1. ${CYAN}./run.sh setup${NC}     # Lần đầu tiên"
    echo -e "  2. ${CYAN}./run.sh fetch${NC}     # Tải data VN30"
    echo -e "  3. ${CYAN}./run.sh dss${NC}       # Xem khuyến nghị"
    echo -e "  4. ${CYAN}./run.sh backtest${NC}  # Kiểm chứng hiệu suất"
}

# ═══════════════ MAIN ═══════════════

case "${1:-help}" in
    setup)    cmd_setup ;;
    fetch)    cmd_fetch ;;
    dss)      cmd_dss ;;
    backtest) cmd_backtest ;;
    test)     cmd_test_phase "$2" ;;
    clean)    cmd_clean ;;
    status)   cmd_status ;;
    help)     cmd_help ;;
    *)
        echo -e "${RED}❌ Lệnh không hợp lệ: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
