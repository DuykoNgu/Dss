#!/bin/bash
# ═══════════════════════════════════════════════════
#  DSS — Script Chạy Nhanh
#  Hệ thống Hỗ trợ Quyết định Mua Bán Cổ Phiếu VN30
# ═══════════════════════════════════════════════════

set -e

# ── Đường dẫn dự án ──
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
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

    echo -e "${YELLOW}📦 Cài thư viện...${NC}"
    pip install --upgrade pip
    pip install -r "$BACKEND_DIR/requirements.txt"

    echo -e "${YELLOW}📁 Tạo thư mục...${NC}"
    mkdir -p "$BACKEND_DIR/data/stocks"
    mkdir -p "$BACKEND_DIR/data/index"
    mkdir -p "$BACKEND_DIR/models"

    echo ""
    echo -e "${GREEN}${BOLD}✅ Setup hoàn tất!${NC}"
    echo -e "   Tiếp theo chạy: ${CYAN}./run.sh fetch${NC} để tải dữ liệu VN30"
}

cmd_fetch() {
    print_header "Tải Dữ Liệu VN30"
    activate_venv

    echo -e "${YELLOW}📡 Đang đồng bộ rổ VN30 (lần đầu ~4 phút do giới hạn quota API)...${NC}"
    python3 "$BACKEND_DIR/main.py" --fetch-only "$@"

    echo ""
    echo -e "${GREEN}${BOLD}✅ Tải dữ liệu hoàn tất!${NC}"
    echo -e "   Dữ liệu lưu tại: ${CYAN}$BACKEND_DIR/data/${NC}"
}

cmd_dss() {
    print_header "Chạy Pipeline DSS"
    activate_venv

    python3 "$BACKEND_DIR/main.py" "$@"
}

cmd_backtest() {
    print_header "Kiểm Chứng Lịch Sử (Backtest)"
    activate_venv

    if [ ! -f "$BACKEND_DIR/backtest_runner.py" ]; then
        echo -e "${RED}❌ Chưa có file backtest_runner.py. Hãy tham khảo README.md để biết cấu trúc và cách chạy.${NC}"
        exit 1
    fi

    python3 "$BACKEND_DIR/backtest_runner.py" "$@"
}

cmd_test() {
    print_header "Unit Test"
    activate_venv
    cd "$BACKEND_DIR" && python3 -m unittest discover -s tests "$@"
}

cmd_web() {
    build_web
    cd "$BACKEND_DIR" && python3 -m api.server "$@"
}

build_web() {
    activate_venv
    if ! command -v npm >/dev/null 2>&1; then
        echo "Cần Node.js và npm để build giao diện React."
        return 1
    fi
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        (cd "$FRONTEND_DIR" && npm ci --no-audit --no-fund) || return 1
    fi
    (cd "$FRONTEND_DIR" && npm run build) || return 1
}

cmd_web_balanced() {
    build_web
    cd "$BACKEND_DIR" || return 1
    python3 -m api.server --build-only || return 1
    python3 -m api.server --producer-only &
    producer_pid=$!
    trap 'kill "$producer_pid" 2>/dev/null || true' EXIT INT TERM
    python3 -m api.server --worker --workers 2 --port 8765
}

cmd_smoke() {
    print_header "Smoke Test Pipeline (offline, 2 mã)"
    activate_venv
    python3 "$BACKEND_DIR/main.py" --no-fetch --limit 2
}

cmd_evaluate() {
    print_header "Đánh Giá ML Walk-Forward"
    activate_venv
    cd "$BACKEND_DIR" && python3 -m src.models.evaluate "$@"
}

cmd_tune() {
    print_header "Tune RF + XGBoost"
    activate_venv
    cd "$BACKEND_DIR" && python3 -m src.models.tune "$@"
}

cmd_clear_data() {
    print_header "Xóa Dữ Liệu Cũ"

    # Hỗ trợ --yes / -y để bỏ qua xác nhận (dùng trong script tự động)
    SKIP_CONFIRM=false
    for arg in "$@"; do
        case "$arg" in
            --yes|-y) SKIP_CONFIRM=true ;;
        esac
    done

    STOCK_COUNT=$(ls "$BACKEND_DIR/data/stocks/"*.csv 2>/dev/null | wc -l | tr -d ' ')
    INDEX_COUNT=$(ls "$BACKEND_DIR/data/index/"*.csv 2>/dev/null | wc -l | tr -d ' ')

    echo -e "   Stocks CSV: ${CYAN}${STOCK_COUNT} file${NC} trong data/stocks/"
    echo -e "   Index CSV:  ${CYAN}${INDEX_COUNT} file${NC} trong data/index/"
    if [ -f "$BACKEND_DIR/data/symbols.json" ]; then
        echo -e "   symbols.json: ${CYAN}có${NC} (sẽ xóa để fetch quét lại rổ VN30 mới)"
    fi
    if [ -f "$BACKEND_DIR/data/market.sqlite3" ]; then
        echo -e "   SQLite: ${CYAN}có${NC} (sẽ xóa cùng dữ liệu CSV cũ)"
    fi
    echo -e "   Models trong ${CYAN}models/${NC}: ${YELLOW}giữ nguyên${NC} (muốn xóa cả models thì dùng './run.sh clean')"
    echo ""

    if [ "$SKIP_CONFIRM" != true ]; then
        echo -ne "${YELLOW}❓ Xóa toàn bộ data cũ? [y/N]: ${NC}"
        read -r CONFIRM
        case "$CONFIRM" in
            [yY][eE][sS]|[yY]) ;;
            *) echo -e "${CYAN}Đã hủy, không xóa gì cả.${NC}"; return 0 ;;
        esac
    fi

    echo -e "${YELLOW}🗑️  Xóa data/stocks/*.csv...${NC}"
    rm -f "$BACKEND_DIR/data/stocks/"*.csv

    echo -e "${YELLOW}🗑️  Xóa data/index/*.csv...${NC}"
    rm -f "$BACKEND_DIR/data/index/"*.csv

    echo -e "${YELLOW}🗑️  Xóa data/symbols.json...${NC}"
    rm -f "$BACKEND_DIR/data/symbols.json"
    rm -f "$BACKEND_DIR/data/market.sqlite3" "$BACKEND_DIR/data/market.sqlite3-wal" "$BACKEND_DIR/data/market.sqlite3-shm"
    rm -f "$BACKEND_DIR/data/web_snapshot.json"
    rm -f "$BACKEND_DIR/data/web_snapshot_sqlite.json"

    mkdir -p "$BACKEND_DIR/data/stocks" "$BACKEND_DIR/data/index"

    echo ""
    echo -e "${GREEN}${BOLD}✅ Đã xóa data cũ!${NC}"
    echo -e "   Tiếp theo chạy: ${CYAN}./run.sh fetch${NC} để tải lại dữ liệu mới"
}

cmd_clean() {
    print_header "Dọn dẹp"

    echo -e "${YELLOW}🗑️  Xóa data cache...${NC}"
    rm -rf "$BACKEND_DIR/data/stocks/"*.csv
    rm -rf "$BACKEND_DIR/data/index/"*.csv
    rm -f "$BACKEND_DIR/data/symbols.json"
    rm -f "$BACKEND_DIR/data/market.sqlite3" "$BACKEND_DIR/data/market.sqlite3-wal" "$BACKEND_DIR/data/market.sqlite3-shm"
    rm -f "$BACKEND_DIR/data/web_snapshot.json"
    rm -f "$BACKEND_DIR/data/web_snapshot_sqlite.json"

    echo -e "${YELLOW}🗑️  Xóa models đã train...${NC}"
    rm -rf "$BACKEND_DIR/models/"*.pkl

    echo -e "${YELLOW}🗑️  Xóa __pycache__...${NC}"
    find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    echo -e "${GREEN}✅ Đã dọn sạch!${NC}"
}

cmd_status() {
    print_header "Trạng Thái Dự Án"

    echo -e "${BOLD}📂 Dữ liệu:${NC}"
    if [ -f "$BACKEND_DIR/data/market.sqlite3" ]; then
        echo -e "   SQLite: ${CYAN}$BACKEND_DIR/data/market.sqlite3${NC}"
        activate_venv
        (cd "$BACKEND_DIR" && python3 -c 'from src.data import store; print("   Phiên mới nhất:", store.latest_date("VNINDEX") or "chưa có")')
    else
        echo -e "   SQLite: ${RED}chưa có (CSV cũ sẽ được nhập khi chạy)${NC}"
    fi

    echo ""
    echo -e "${BOLD}🤖 Models:${NC}"
    if [ -d "$BACKEND_DIR/models" ]; then
        BUNDLE_COUNT=$(ls "$BACKEND_DIR/models/"*_bundle.pkl 2>/dev/null | wc -l | tr -d ' ')
        LEGACY_COUNT=$(ls "$BACKEND_DIR/models/"*_rf.pkl "$BACKEND_DIR/models/"*_xgb.pkl 2>/dev/null | wc -l | tr -d ' ')
        echo -e "   Bundles:    ${CYAN}${BUNDLE_COUNT}${NC} · file cũ: ${YELLOW}${LEGACY_COUNT}${NC}"
    else
        echo -e "   Trained:    ${RED}chưa có${NC}"
    fi

    echo ""
    echo -e "${BOLD}📝 Source files:${NC}"
    for f in pipeline data/data_fetcher data/data_cleaner features/indicators features/features models/ml_models scoring/scoring scoring/decision backtest/backtester; do
        if [ -f "$BACKEND_DIR/src/${f}.py" ]; then
            echo -e "   src/${f}.py   ${GREEN}✅${NC}"
        else
            echo -e "   src/${f}.py   ${RED}❌ chưa tạo${NC}"
        fi
    done

    for f in main.py backtest_runner.py; do
        if [ -f "$BACKEND_DIR/$f" ]; then
            echo -e "   ${f}            ${GREEN}✅${NC}"
        else
            echo -e "   ${f}            ${RED}❌ chưa tạo${NC}"
        fi
    done
}

cmd_push() {
    print_header "Push Code Lên GitHub"

    cd "$PROJECT_DIR" || exit 1

    # ── 1. Kiểm tra git repo ──
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        echo -e "${RED}❌ Chưa phải git repo. Chạy: git init${NC}"
        exit 1
    fi

    BRANCH=$(git branch --show-current)
    [ -z "$BRANCH" ] && BRANCH="main"

    # ── 2. Gom message từ tất cả tham số ──
    # Dùng: ./run.sh push "fix scoring"  hoặc  ./run.sh push (tự sinh)
    MSG="$*"
    if [ -z "$MSG" ]; then
        MSG="update: $(date '+%Y-%m-%d %H:%M')"
    fi

    echo -e "   Branch:  ${CYAN}${BRANCH}${NC}"
    echo -e "   Message: ${CYAN}${MSG}${NC}"
    echo ""

    # ── 3. Add tất cả (tôn trọng .gitignore → data/, models/, .venv tự bỏ qua) ──
    echo -e "${YELLOW}📦 git add...${NC}"
    git add -A

    # ── 4. Nếu không có gì để commit → kiểm tra có commit chưa push không ──
    if git diff --cached --quiet; then
        echo -e "${YELLOW}ℹ️  Không có thay đổi mới để commit.${NC}"
        AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)
        if [ "$AHEAD" != "0" ] && [ -n "$AHEAD" ]; then
            echo -e "${YELLOW}📤 Có ${AHEAD} commit chưa push → đang push...${NC}"
        else
            echo -e "${GREEN}✅ Mọi thứ đã đồng bộ, không cần push.${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}📝 git commit...${NC}"
        git commit -m "$MSG"
    fi

    echo ""
    echo -e "${YELLOW}🔄 Pull --rebase để tránh conflict...${NC}"
    if ! git pull --rebase origin "$BRANCH" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Không pull được (có thể remote chưa có branch này hoặc mất mạng). Bỏ qua, tiếp tục push...${NC}"
    fi

    echo ""
    echo -e "${YELLOW}📤 git push...${NC}"
    if git rev-parse --abbrev-ref --symbolic-full-name @{u} &>/dev/null; then
        git push
    else
        git push -u origin "$BRANCH"
    fi

    echo ""
    echo -e "${GREEN}${BOLD}✅ Push hoàn tất!${NC}"
    echo -e "   Xem tại: ${CYAN}$(git remote get-url origin 2>/dev/null)${NC}"
}

cmd_help() {
    print_header "Hướng Dẫn Sử Dụng"

    echo -e "${BOLD}Cú pháp:${NC} ./run.sh <lệnh>"
    echo ""
    echo -e "${BOLD}Các lệnh:${NC}"
    echo -e "  ${CYAN}setup${NC}        Tạo venv, cài thư viện, tạo thư mục"
    echo -e "  ${CYAN}fetch${NC}        Đồng bộ VN30: mã thiếu tải full, mã cũ tải chồng 10 ngày cuối"
    echo -e "  ${CYAN}dss${NC}          Chạy khuyến nghị hôm nay (main.py)"
    echo -e "  ${CYAN}dss --retrain${NC} Ép train lại model (bình thường tự train lại khi model cũ)"
    echo -e "  ${CYAN}backtest${NC}     Backtest so sánh blend/rule/ml (thêm --pooled để dùng model chung)"
    echo -e "  ${CYAN}test${NC}         Chạy unit test"
    echo -e "  ${CYAN}smoke${NC}        Smoke test pipeline (offline, 2 mã)"
    echo -e "  ${CYAN}evaluate${NC}     Label distribution + metrics + confusion matrix"
    echo -e "  ${CYAN}tune${NC}         So sánh cấu hình RF/XGBoost bằng walk-forward"
    echo -e "  ${CYAN}web${NC}          Build React và mở bảng giá VN30 tại localhost:8765"
    echo -e "  ${CYAN}web-balanced${NC} Build React, chạy producer và 2 ASGI worker tại localhost:8765"
    echo -e "  ${CYAN}status${NC}       Kiểm tra trạng thái dự án (data, models, files)"
    echo -e "  ${CYAN}clear-data${NC}   Xóa SQLite và CSV cũ, giữ models"
    echo -e "  ${CYAN}clean${NC}        Xóa cache data, models, __pycache__"
    echo -e "  ${CYAN}push${NC}         Push code lên GitHub bằng 1 lệnh"
    echo -e "  ${CYAN}help${NC}         Hiển thị hướng dẫn này"
    echo ""
    echo -e "${BOLD}Quy trình đề xuất:${NC}"
    echo -e "  1. ${CYAN}./run.sh setup${NC}     # Lần đầu tiên"
    echo -e "  2. ${CYAN}./run.sh fetch${NC}     # Tải data VN30"
    echo -e "  3. ${CYAN}./run.sh dss${NC}       # Xem khuyến nghị"
    echo -e "  4. ${CYAN}./run.sh backtest${NC}  # Kiểm chứng hiệu suất"
    echo ""
    echo -e "${BOLD}Push code:${NC}"
    echo -e "  ${CYAN}./run.sh push \"mô tả thay đổi\"${NC}"
    echo -e "  ${CYAN}./run.sh push${NC}  # tự tạo message theo ngày giờ"
}

# ═══════════════ MAIN ═══════════════

case "${1:-help}" in
    setup)    cmd_setup ;;
    fetch)    shift; cmd_fetch "$@" ;;
    dss)      shift; cmd_dss "$@" ;;
    backtest) shift; cmd_backtest "$@" ;;
    test)     shift; cmd_test "$@" ;;
    smoke)    cmd_smoke ;;
    evaluate) shift; cmd_evaluate "$@" ;;
    tune)     shift; cmd_tune "$@" ;;
    web)      shift; cmd_web "$@" ;;
    web-balanced) cmd_web_balanced ;;
    clear-data) cmd_clear_data "$2" ;;
    clean)    cmd_clean ;;
    status)   cmd_status ;;
    push)     shift; cmd_push "$@" ;;
    help)     cmd_help ;;
    *)
        echo -e "${RED}❌ Lệnh không hợp lệ: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
