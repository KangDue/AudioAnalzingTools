import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

# 상위 경로에서 모듈 import 가능하록 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from custom_region_widget import CustomRegionWidget
from layout_builder import LayoutBuilder, LayoutState

SPEC = {
    "type": "split",
    "orientation": "h",
    "sizes": [2, 3],
    "children": [
        {
            "type": "region",
            "name": "left",
            "widgets": [
                ("label", lambda: QLabel("왼쪽 패널")),
                ("btn", lambda: QPushButton("왼쪽 버튼")),
            ],
        },
        {
            "type": "split",
            "orientation": "v",
            "children": [
                {
                    "type": "region",
                    "name": "right_top",
                    "widgets": [("label", lambda: QLabel("오른쪽 위"))],
                },
                {
                    "type": "region",
                    "name": "right_bottom",
                    "widgets": [("label", lambda: QLabel("오른쪽 아래"))],
                },
            ],
        },
    ],
}

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("선언적 레이아웃 빌더 예제")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        self.builder = LayoutBuilder(CustomRegionWidget)
        self.root = self.builder.build(SPEC)
        layout.addWidget(self.root)

        # 상호작용 예시
        left_btn = self.builder.registry["left.btn"]
        right_bottom_label = self.builder.registry["right_bottom.label"]
        left_btn.clicked.connect(lambda: right_bottom_label.setText("왼쪽 버튼에서 변경!"))

        # 상태 복원
        self.state = LayoutState(org="UiCustom", app="DeclarativeDemo")
        self.state.restore(self.root)

    def closeEvent(self, event):
        # 상태 저장
        self.state.save(self.root)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())