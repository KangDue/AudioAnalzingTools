import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTabWidget, QHBoxLayout
from PyQt5.QtCore import Qt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from layout_builder import LayoutBuilder, LayoutState
from custom_region_widget import CustomRegionWidget
from PyQt5.QtWidgets import QLabel

SPEC = {
    "type": "tabs",
    "tabs": [
        {"title": "첫번째", "content": {"type": "region", "name": "tab1", "widgets": [("l", lambda: QLabel("탭1 내용"))]}},
        {"title": "두번째", "content": {"type": "region", "name": "tab2", "widgets": [("l", lambda: QLabel("탭2 내용"))]}},
    ],
}

class TabsDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("탭 관리 예제")
        self.resize(800, 600)
        self.layout = QVBoxLayout(self)

        self.builder = LayoutBuilder(CustomRegionWidget)
        self.root = self.builder.build(SPEC)
        self.layout.addWidget(self.root)

        self.tabs = self.root.findChild(QTabWidget)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.remove_tab)

        self.state = LayoutState(org="UiCustom", app="TabsDemo")
        self.state.restore(self.tabs)

        control_layout = QHBoxLayout()
        add_btn = QPushButton("탭 추가")
        add_btn.clicked.connect(self.add_tab)
        control_layout.addWidget(add_btn)
        self.layout.addLayout(control_layout)

    def add_tab(self):
        count = self.tabs.count() + 1
        new_content = CustomRegionWidget()
        new_content.add(QLabel(f"새 탭 {count} 내용"))
        self.tabs.addTab(new_content, f"새 탭 {count}")

    def remove_tab(self, index):
        if self.tabs.count() > 1:
            widget = self.tabs.widget(index)
            self.tabs.removeTab(index)
            widget.deleteLater()

    def closeEvent(self, event):
        self.state.save(self.tabs)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TabsDemo()
    win.show()
    sys.exit(app.exec_())