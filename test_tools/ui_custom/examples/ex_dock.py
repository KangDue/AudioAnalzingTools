import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QTextEdit, QLabel

# 상위 경로에서 모듈 import 가능하도록 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from layout_builder import LayoutBuilder, LayoutState
from custom_region_widget import CustomRegionWidget

SPEC = {
    "type": "main_window",
    "title": "도킹 레이아웃 선언적 예제",
    "central": {
        "type": "region",
        "name": "editor",
        "widgets": [
            ("text", lambda: QTextEdit("// 중앙 에디터")),
        ],
    },
    "docks": [
        {
            "name": "project",
            "title": "Project",
            "area": "left",
            "features": ["closable", "movable", "floatable"],
            "allowedAreas": ["left", "right"],
            "content": {
                "type": "region",
                "name": "project_region",
                "widgets": [("label", lambda: QLabel("프로젝트 트리(샘플)"))],
            },
        },
        {
            "name": "console",
            "title": "Console",
            "area": "bottom",
            "features": ["closable", "movable", "floatable"],
            "allowedAreas": ["bottom", "top"],
            "content": {
                "type": "region",
                "name": "console_region",
                "widgets": [("log", lambda: QTextEdit("> 로그 출력..."))],
            },
            "tabify_with": ["project"],
        },
    ],
}

if __name__ == "__main__":
    app = QApplication(sys.argv)
    builder = LayoutBuilder(CustomRegionWidget)
    root = builder.build(SPEC)
    state = LayoutState(org="UiCustom", app="DockingDemo")
    state.restore(root)
    root.show()
    try:
        sys.exit(app.exec_())
    finally:
        state.save(root)