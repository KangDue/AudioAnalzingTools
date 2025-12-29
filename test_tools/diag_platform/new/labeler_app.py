import sys
import os

# [핵심] PyQt5 강제 사용 설정 (PySide6와의 충돌 방지)
os.environ["QT_API"] = "pyqt5"

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor

import sounddevice as sd
import pyqtgraph as pg
from scipy import signal
import numpy as np
from backend import DataManager


# ... 기존 import 아래에 추가 ...
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class LabelerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Labeler (라벨링 도구)")
        self.resize(1400, 900)
        self.db_manager = DataManager()
        
        self.init_style()
        self.labeler_tab = DataLabelerTab(self.db_manager)
        self.setCentralWidget(self.labeler_tab)

    def init_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-size: 14px; }
            QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QTableWidget { background-color: #1e1e1e; color: #fff; gridline-color: #444; selection-background-color: #0078D7; }
            QHeaderView::section { background-color: #444; color: #fff; padding: 4px; }
            QListWidget { background-color: #1e1e1e; color: #fff; border: 1px solid #555; }
            QListWidget::item:selected { background-color: #0078D7; color: white; border: 1px solid #005A9E; }
            QListWidget::item:hover { background-color: #333333; }
            QPushButton { background-color: #555; color: #fff; border: 1px solid #777; border-radius: 3px; padding: 6px; }
            QPushButton:hover { background-color: #666; }
            QTextEdit, QLineEdit, QSpinBox { background-color: #1e1e1e; color: #fff; border: 1px solid #555; }
            QComboBox { background-color: #1e1e1e; color: white; border: 1px solid #555; padding: 4px; }
            QComboBox QAbstractItemView { background-color: #1e1e1e; color: white; selection-background-color: #0078D7; }
            QMessageBox { background-color: #2b2b2b; }
            QMessageBox QLabel { color: #ffffff; background-color: transparent; }
            QMessageBox QPushButton { min-width: 60px; }
            /* Splitter Handle 스타일 (잘 보이게) */
            QSplitter::handle { background-color: #444; }
            QSplitter::handle:hover { background-color: #0078D7; }
        """)

class ClusteringDialog(QDialog):
    def __init__(self, parent=None, db_manager=None, filtered_ids=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Labeling (Clustering)")
        self.resize(500, 600)
        self.db = db_manager
        self.target_ids = filtered_ids
        self.generated_labels = {} # {id: new_label}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 1. Info
        info_grp = QGroupBox("Target Data Info")
        info_layout = QVBoxLayout()
        self.lbl_count = QLabel(f"Target Items: {len(self.target_ids)} items (Filtered Results)")
        info_layout.addWidget(self.lbl_count)
        info_grp.setLayout(info_layout)
        layout.addWidget(info_grp)
        
        # 2. Algorithm Settings
        algo_grp = QGroupBox("Clustering Algorithm")
        algo_layout = QFormLayout()
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["K-Means", "DBSCAN", "Agglomerative (Hierarchical)"])
        self.combo_algo.currentIndexChanged.connect(self.update_param_ui)
        
        self.spin_param1 = QSpinBox() # Clusters or Epsilon
        self.spin_param2 = QSpinBox() # Min Samples (for DBSCAN)
        self.lbl_param1 = QLabel("Number of Clusters (k):")
        self.lbl_param2 = QLabel("Min Samples:")
        
        algo_layout.addRow("Algorithm:", self.combo_algo)
        algo_layout.addRow(self.lbl_param1, self.spin_param1)
        algo_layout.addRow(self.lbl_param2, self.spin_param2)
        
        algo_grp.setLayout(algo_layout)
        layout.addWidget(algo_grp)
        
        # 3. Labeling Strategy
        lbl_grp = QGroupBox("Labeling Strategy")
        lbl_layout = QFormLayout()
        
        self.combo_target = QComboBox()
        self.combo_target.addItems(["label_mid", "label_bot"]) # Top은 보통 명확해서 자동화 위험
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Overwrite (Replace)", "Append (+)", "Custom Name + Cluster ID"])
        
        self.txt_custom = QLineEdit()
        self.txt_custom.setPlaceholderText("e.g. Group_")
        
        lbl_layout.addRow("Target Column:", self.combo_target)
        lbl_layout.addRow("Naming Mode:", self.combo_mode)
        lbl_layout.addRow("Custom Prefix:", self.txt_custom)
        
        lbl_grp.setLayout(lbl_layout)
        layout.addWidget(lbl_grp)
        
        # 4. Action
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run Clustering & Apply")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_clustering)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        # 초기 UI 설정
        self.update_param_ui()

    def update_param_ui(self):
        algo = self.combo_algo.currentText()
        if algo == "K-Means" or algo == "Agglomerative (Hierarchical)":
            self.lbl_param1.setText("Number of Clusters (k):")
            self.spin_param1.setRange(2, 50)
            self.spin_param1.setValue(5)
            self.lbl_param2.setVisible(False)
            self.spin_param2.setVisible(False)
        elif algo == "DBSCAN":
            self.lbl_param1.setText("Epsilon (distance):")
            self.spin_param1.setRange(1, 1000) # Scaled distance assumed
            self.spin_param1.setValue(5) # Default eps (needs tuning based on scaling)
            
            self.lbl_param2.setVisible(True)
            self.spin_param2.setVisible(True)
            self.lbl_param2.setText("Min Samples:")
            self.spin_param2.setRange(2, 50)
            self.spin_param2.setValue(5)

    def run_clustering(self):
        if not self.target_ids:
            QMessageBox.warning(self, "Error", "No data to cluster.")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Loading Data & Processing...")
        QApplication.processEvents() # UI 갱신
        
        try:
            # 1. Feature Load
            X, valid_ids = self.db.get_features_for_clustering(self.target_ids)
            if X is None or len(X) == 0:
                raise Exception("Failed to load features. Ensure H5 file exists.")
            
            # 2. Preprocessing (Impute & Scale)
            # 결측치(NaN)가 있으면 평균으로 채움
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # 스케일링 (매우 중요: SPL은 60~90인데 계수는 0~1이면 거리 계산 망함)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 3. Clustering
            algo_name = self.combo_algo.currentText()
            labels = []
            
            if algo_name == "K-Means":
                k = self.spin_param1.value()
                model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = model.fit_predict(X_scaled)
                
            elif algo_name == "DBSCAN":
                # DBSCAN의 eps는 데이터 분포에 민감함. UI에서 받은 값은 정수지만 여기선 0.1 곱해서 미세조정 가정
                eps = self.spin_param1.value() * 0.1 
                min_samples = self.spin_param2.value()
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                
            elif algo_name == "Agglomerative (Hierarchical)":
                k = self.spin_param1.value()
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X_scaled)
            
            # 4. Generate New Labels
            target_col = self.combo_target.currentText()
            mode = self.combo_mode.currentText()
            prefix = self.txt_custom.text().strip()
            
            update_map = {} # {id: new_label_text}
            
            # 기존 라벨 정보를 가져와야 하는 경우 (Append 모드)
            existing_labels = {}
            if "Append" in mode:
                # 쿼리로 현재 라벨 가져오기
                self.db.cursor.execute(f"SELECT id, {target_col} FROM noise_data WHERE id IN ({','.join(['?']*len(valid_ids))})", valid_ids)
                existing_labels = dict(self.db.cursor.fetchall())

            for uid, cluster_id in zip(valid_ids, labels):
                cluster_str = f"C{cluster_id}" if cluster_id != -1 else "Noise" # DBSCAN -1 is noise
                new_val = ""
                
                if "Custom Name" in mode:
                    if prefix: new_val = f"{prefix}_{cluster_str}"
                    else: new_val = f"Auto_{cluster_str}"
                elif "Append" in mode:
                    old_val = existing_labels.get(uid, "")
                    if old_val and old_val != "None":
                        new_val = f"{old_val}+{cluster_str}"
                    else:
                        new_val = cluster_str
                else: # Overwrite
                    new_val = f"Cluster_{cluster_str}"
                
                update_map[uid] = new_val
            
            # 5. DB Update
            updated_count = self.db.update_labels_from_dict(update_map, target_col)
            
            # 6. Auto-Add new labels to settings (backend logic handles this? No, we need explicit add)
            # 새로 생긴 라벨들을 label_settings에 등록
            unique_new_labels = set(update_map.values())
            category_key = 'mid' if target_col == 'label_mid' else 'bot'
            for lbl in unique_new_labels:
                self.db.ensure_label_exists(category_key, lbl)
            
            QMessageBox.information(self, "Success", f"Clustering Completed.\nAlgorithm: {algo_name}\nUpdated Items: {updated_count}")
            self.accept() # 다이얼로그 닫기
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.btn_run.setText("▶ Run Clustering & Apply")
            self.btn_run.setEnabled(True)

class DataLabelerTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        
        self.settings = QSettings("FactoryLabeler", "GraphSettings")
        
        self.current_page = 1
        self.items_per_page = 20
        self.total_items = 0
        self.total_pages = 1
        
        self.current_id = None
        self.current_sr = 51200
        self.current_data = {}
        self.current_feat_arr = []
        self.current_feat_names = []
        self.is_loading_table = False 
        
        self.top_labels = ["Bearing", "Others"]
        self.mid_labels = ["Normal", "Noise", "Vibration"]
        self.bot_labels = ["OK", "NG", "Unknown"]
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # --- Zone 1: DB & Search ---
        self.zone1 = QGroupBox("Zone 1: DB & Search")
        z1_layout = QVBoxLayout()
        
        db_layout = QHBoxLayout()
        self.btn_open_db = QPushButton("📂 Open DB")
        self.btn_open_db.clicked.connect(self.open_db_file)
        self.lbl_db_name = QLabel("No DB Loaded")
        self.lbl_db_name.setStyleSheet("color: #aaa; font-size: 11px;")
        db_layout.addWidget(self.btn_open_db); db_layout.addWidget(self.lbl_db_name)
        z1_layout.addLayout(db_layout)
        
        search_grp = QGroupBox("Search Filters")
        search_grid = QGridLayout()
        self.chk_date_enable = QCheckBox("Date Filter:")
        search_grid.addWidget(self.chk_date_enable, 0, 0)
        date_layout = QHBoxLayout()
        self.date_from = QDateEdit(); self.date_from.setCalendarPopup(True); self.date_from.setDisplayFormat("yyyy-MM-dd"); self.date_from.setDate(QDate.currentDate().addMonths(-1))
        self.date_to = QDateEdit(); self.date_to.setCalendarPopup(True); self.date_to.setDisplayFormat("yyyy-MM-dd"); self.date_to.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("From")); date_layout.addWidget(self.date_from); date_layout.addWidget(QLabel("To")); date_layout.addWidget(self.date_to)
        search_grid.addLayout(date_layout, 0, 1)
        search_grid.addWidget(QLabel("Serial:"), 1, 0)
        self.txt_serial = QLineEdit(); self.txt_serial.setPlaceholderText("ex) R1K6...")
        search_grid.addWidget(self.txt_serial, 1, 1)
        search_grid.addWidget(QLabel("Top:"), 2, 0); self.combo_filter_top = QComboBox(); search_grid.addWidget(self.combo_filter_top, 2, 1)
        search_grid.addWidget(QLabel("Mid:"), 3, 0); self.combo_filter_mid = QComboBox(); search_grid.addWidget(self.combo_filter_mid, 3, 1)
        search_grid.addWidget(QLabel("Bot:"), 4, 0); self.combo_filter_bot = QComboBox(); search_grid.addWidget(self.combo_filter_bot, 4, 1)
        search_grid.addWidget(QLabel("Status:"), 5, 0); self.combo_status = QComboBox()
        self.combo_status.addItems(["All", "Unlabeled", "Labeled"])
        search_grid.addWidget(self.combo_status, 5, 1)
        logic_layout = QHBoxLayout()
        self.rb_and = QRadioButton("AND"); self.rb_or = QRadioButton("OR"); self.rb_and.setChecked(True)
        logic_layout.addWidget(self.rb_and); logic_layout.addWidget(self.rb_or)
        search_grid.addLayout(logic_layout, 6, 0, 1, 2)
        self.btn_search = QPushButton("🔍 Apply Filter"); self.btn_search.clicked.connect(self.reset_and_load)
        search_grid.addWidget(self.btn_search, 7, 0, 1, 1)#7, 0, 1, 2
        
        # [신규] 오토 라벨링 버튼 추가
        self.btn_auto = QPushButton("🤖 Auto Label (Clustering)")
        self.btn_auto.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold;")
        self.btn_auto.clicked.connect(self.open_clustering_dialog)
        search_grid.addWidget(self.btn_auto, 7, 1, 1, 1)   # 옆에 배치
        
        search_grp.setLayout(search_grid); z1_layout.addWidget(search_grp)

        self.table = QTableWidget(); self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Bot", "Mid", "Top", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.on_table_click)
        self.table.currentCellChanged.connect(self.on_current_cell_changed)
        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.installEventFilter(self)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        z1_layout.addWidget(self.table)
        
        page_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(self.go_prev_page)
        self.lbl_page = QLabel("0 / 0"); self.lbl_page.setAlignment(Qt.AlignCenter); self.lbl_page.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(self.go_next_page)
        page_layout.addWidget(self.btn_prev); page_layout.addWidget(self.lbl_page); page_layout.addWidget(self.btn_next)
        z1_layout.addLayout(page_layout); self.zone1.setLayout(z1_layout)

        # --- Zone 2: Visualization ---
        self.zone2 = QGroupBox("Zone 2: Visualization")
        z2_layout = QVBoxLayout()
        chk_style = "QCheckBox { color: black; background-color: #dddddd; padding: 4px; border-radius: 4px; font-weight: bold; }"
        
        ctrl_layout = QHBoxLayout()
        self.chk_ch1 = QCheckBox("Ch1"); self.chk_ch1.setChecked(True); self.chk_ch1.setStyleSheet(chk_style)
        self.chk_ch2 = QCheckBox("Ch2"); self.chk_ch2.setChecked(True); self.chk_ch2.setStyleSheet(chk_style)
        self.chk_ch3 = QCheckBox("Ch3"); self.chk_ch3.setChecked(True); self.chk_ch3.setStyleSheet(chk_style)
        self.chk_ch4 = QCheckBox("Ch4"); self.chk_ch4.setStyleSheet(chk_style)
        self.chk_ch1.stateChanged.connect(self.update_plots); self.chk_ch2.stateChanged.connect(self.update_plots)
        self.chk_ch3.stateChanged.connect(self.update_plots); self.chk_ch4.stateChanged.connect(self.update_plots)
        ctrl_layout.addWidget(self.chk_ch1); ctrl_layout.addWidget(self.chk_ch2)
        ctrl_layout.addWidget(self.chk_ch3); ctrl_layout.addWidget(self.chk_ch4); z2_layout.addLayout(ctrl_layout)
        
        ctrl_layout2 = QHBoxLayout()
        self.chk_sync = QCheckBox("Sync X"); self.chk_sync.setStyleSheet(chk_style)
        self.chk_sync.stateChanged.connect(self.update_plots)
        lbl_cmap = QLabel("Color:")
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['inferno', 'viridis', 'plasma', 'magma', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'])
        saved_cmap = self.settings.value("colormap", "inferno", type=str)
        self.combo_cmap.setCurrentText(saved_cmap)
        self.combo_cmap.currentTextChanged.connect(self.on_setting_changed)
        
        lbl_min = QLabel("Min Hz:")
        self.spin_min = QSpinBox(); self.spin_min.setRange(0, 25600); self.spin_min.setSingleStep(100); self.spin_min.setSuffix(" Hz")
        lbl_max = QLabel("Max Hz:")
        self.spin_max = QSpinBox(); self.spin_max.setRange(100, 25600); self.spin_max.setSingleStep(100); self.spin_max.setSuffix(" Hz")
        self.spin_min.setValue(self.settings.value("min_freq", 0, type=int))
        self.spin_max.setValue(self.settings.value("max_freq", 25600, type=int))
        self.spin_min.valueChanged.connect(self.on_setting_changed); self.spin_max.valueChanged.connect(self.on_setting_changed)

        self.btn_feat = QPushButton("View Feature"); self.btn_feat.clicked.connect(self.show_feature_popup)
        ctrl_layout2.addWidget(self.chk_sync); ctrl_layout2.addSpacing(10); ctrl_layout2.addWidget(lbl_cmap); ctrl_layout2.addWidget(self.combo_cmap)
        ctrl_layout2.addSpacing(10); ctrl_layout2.addWidget(lbl_min); ctrl_layout2.addWidget(self.spin_min); ctrl_layout2.addWidget(lbl_max); ctrl_layout2.addWidget(self.spin_max)
        ctrl_layout2.addStretch(1); ctrl_layout2.addWidget(self.btn_feat)
        z2_layout.addLayout(ctrl_layout2)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setSpacing(20)
        self.plot_container.setLayout(self.plot_layout)
        self.scroll_area.setWidget(self.plot_container)
        z2_layout.addWidget(self.scroll_area)
        self.zone2.setLayout(z2_layout)

        # --- Zone 3,4,5: Labels (with Splitter) ---
        self.zone_labels = QWidget()
        label_layout = QHBoxLayout()
        self.zone_labels.setLayout(label_layout)
        
        def create_label_zone(title, label_col, category_key):
            grp = QGroupBox(title)
            main_v = QVBoxLayout() # 그룹박스 메인 레이아웃
            grp.setLayout(main_v)
            
            # 1. 상단 버튼 (고정)
            btn_box = QHBoxLayout()
            btn_add = QPushButton("➕"); btn_add.setToolTip("Add Label")
            btn_edit = QPushButton("✎"); btn_edit.setToolTip("Rename Label")
            btn_del = QPushButton("➖"); btn_del.setToolTip("Delete Label")
            for b in [btn_add, btn_edit, btn_del]: b.setFixedWidth(40)
            btn_box.addWidget(btn_add); btn_box.addWidget(btn_edit); btn_box.addWidget(btn_del)
            main_v.addLayout(btn_box)

            # 2. 스플리터 (Vertical)
            splitter = QSplitter(Qt.Vertical)
            
            # 2-1. 라벨 리스트 (위)
            lst = QListWidget()
            lst.setDragDropMode(QAbstractItemView.InternalMove)
            # [중요] 고정 높이 제거
            splitter.addWidget(lst) 
            
            # 2-2. 미리보기 영역 (아래) - 컨테이너 위젯으로 묶음
            preview_container = QWidget()
            preview_layout = QVBoxLayout()
            preview_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_prev = QLabel("▼ Comparison:")
            lbl_prev.setStyleSheet("color: #888; font-size: 10px; margin-top: 5px;")
            lst_prev = QListWidget()
            lst_prev.setStyleSheet("font-size: 11px; color: #ddd; background-color: #222;")
            
            preview_layout.addWidget(lbl_prev)
            preview_layout.addWidget(lst_prev)
            preview_container.setLayout(preview_layout)
            
            splitter.addWidget(preview_container)
            
            # 초기 비율 설정 (상단 60%, 하단 40% 정도)
            splitter.setSizes([200, 100])
            
            main_v.addWidget(splitter)

            # 이벤트 연결
            lst.model().rowsMoved.connect(lambda: self.save_label_order(category_key, lst))
            btn_add.clicked.connect(lambda: self.add_new_label(lst, category_key))
            btn_edit.clicked.connect(lambda: self.rename_label_action(lst, label_col, category_key))
            btn_del.clicked.connect(lambda: self.delete_label(lst, label_col, category_key))
            lst.itemClicked.connect(lambda item: self.load_preview(label_col, item.text(), lst_prev))
            lst_prev.itemClicked.connect(lambda item: self.load_data(item.text()))
            
            return grp, lst, lst_prev

        self.z3_grp, self.list_bot, self.prev_bot = create_label_zone("Zone 3: Bottom", "label_bot", "bot")
        self.z4_grp, self.list_mid, self.prev_mid = create_label_zone("Zone 4: Mid", "label_mid", "mid")
        self.z5_grp, self.list_top, self.prev_top = create_label_zone("Zone 5: Top", "label_top", "top")
        
        self.btn_save = QPushButton("💾 Set Label (Save & Next)")
        self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_and_next)
        
        self.btn_bulk = QPushButton("⚡ Batch Apply to Search Results")
        self.btn_bulk.setStyleSheet("background-color: #F57C00; color: white; font-weight: bold; padding: 8px;")
        self.btn_bulk.clicked.connect(self.bulk_update_action)
        
        self.btn_export = QPushButton("📤 Export Data")
        self.btn_export.clicked.connect(self.export_data)
        
        self.z5_grp.layout().addWidget(self.btn_save)
        self.z5_grp.layout().addWidget(self.btn_bulk)
        self.z5_grp.layout().addWidget(self.btn_export)
        
        label_layout.addWidget(self.z3_grp); label_layout.addWidget(self.z4_grp); label_layout.addWidget(self.z5_grp)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.zone1); splitter.addWidget(self.zone2); splitter.addWidget(self.zone_labels)
        splitter.setSizes([350, 600, 350])
        main_layout.addWidget(splitter)

    # --- Event Filters ---
    def eventFilter(self, source, event):
        if source == self.table and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Down:
                current_row = self.table.currentRow()
                if current_row == self.table.rowCount() - 1 and self.current_page < self.total_pages:
                    self.go_next_page()
                    self.table.selectRow(0); self.on_table_click(0, 0)
                    return True
            elif event.key() == Qt.Key_Up:
                current_row = self.table.currentRow()
                if current_row == 0 and self.current_page > 1:
                    self.go_prev_page()
                    last_row = self.table.rowCount() - 1
                    self.table.selectRow(last_row); self.on_table_click(last_row, 0)
                    return True
        return super().eventFilter(source, event)

    # --- Handlers ---
    def on_current_cell_changed(self, current_row, current_col, previous_row, previous_col):
        if self.is_loading_table or current_row < 0: return
        item = self.table.item(current_row, 0)
        if item: self.load_data(item.text())

    def on_table_click(self, row, col):
        item = self.table.item(row, 0)
        if item: self.load_data(item.text())

    def on_table_item_changed(self, item):
        if self.is_loading_table or item.column() in [0, 4]: return
        uid = self.table.item(item.row(), 0).text()
        new_val = item.text().strip()
        col_map = {1: ('label_bot', 'bot'), 2: ('label_mid', 'mid'), 3: ('label_top', 'top')}
        if item.column() not in col_map: return
        db_col, category = col_map[item.column()]
        if self.db.update_single_label(uid, db_col, new_val):
            if new_val and self.db.ensure_label_exists(category, new_val): self.load_labels_from_db()
            top_item = self.table.item(item.row(), 3)
            has_top = top_item and top_item.text().strip() != ""
            status_item = self.table.item(item.row(), 4)
            if has_top: status_item.setText("Labeled"); status_item.setForeground(QColor("#00FF00"))
            else: status_item.setText("Unlabeled"); status_item.setForeground(QColor("#FFaa00"))
            self.refresh_stats()

    # --- Setting Handler ---
    def on_setting_changed(self):
        self.settings.setValue("colormap", self.combo_cmap.currentText())
        self.settings.setValue("min_freq", self.spin_min.value())
        self.settings.setValue("max_freq", self.spin_max.value())
        self.update_plots()

    # --- Label Management Actions ---
    def add_new_label(self, list_widget, category):
        text, ok = QInputDialog.getText(self, "Add Label", "New Label Name:")
        if ok and text:
            items = [list_widget.item(x).text().split(' (')[0] for x in range(list_widget.count())]
            if text in items: QMessageBox.warning(self, "Duplicate", "Label already exists."); return
            list_widget.addItem(text); self.save_label_order(category, list_widget); self.update_combos()

    def rename_label_action(self, list_widget, label_col, category):
        item = list_widget.currentItem()
        if not item: QMessageBox.warning(self, "Select", "Please select a label to edit."); return
        old_name = item.text().split(' (')[0]
        new_name, ok = QInputDialog.getText(self, "Rename Label", f"Rename '{old_name}' to:", text=old_name)
        if ok and new_name and new_name != old_name:
            items = [list_widget.item(x).text().split(' (')[0] for x in range(list_widget.count())]
            if new_name in items: QMessageBox.warning(self, "Duplicate", "Name exists."); return
            if self.db.rename_label(category, label_col, old_name, new_name):
                self.load_labels_from_db(); self.refresh_stats(); self.load_list()
                QMessageBox.information(self, "Success", f"Renamed to '{new_name}'.")
            else: QMessageBox.critical(self, "Error", "Failed to rename.")

    def delete_label(self, list_widget, label_col, category):
        item = list_widget.currentItem()
        if not item: QMessageBox.warning(self, "Select", "Please select a label to delete."); return
        label_text = item.text().split(' (')[0]
        reply = QMessageBox.question(self, "Delete Label", f"Delete '{label_text}'?\nRelated data will become Unlabeled.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.db.remove_label_from_records(label_col, label_text):
                list_widget.takeItem(list_widget.row(item)); self.save_label_order(category, list_widget)
                self.refresh_stats(); self.update_combos(); self.load_list()
                QMessageBox.information(self, "Done", "Label deleted.")
            else: QMessageBox.critical(self, "Error", "DB Update Failed.")

    def bulk_update_action(self):
        top = self.list_top.currentItem().text().split(' (')[0] if self.list_top.currentItem() else None
        mid = self.list_mid.currentItem().text().split(' (')[0] if self.list_mid.currentItem() else None
        bot = self.list_bot.currentItem().text().split(' (')[0] if self.list_bot.currentItem() else None
        if not top: QMessageBox.warning(self, "Warning", "Select Top Label!"); return
        msg = f"Apply to ALL {self.total_items} items?\nTop: {top}\nMid: {mid}\nBot: {bot}\nUndonable!"
        if QMessageBox.question(self, "Batch Apply", msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            opts = self.get_filter_opts(); logic = self.get_logic_operator()
            cnt = self.db.bulk_update_labels(opts, logic, top, mid, bot)
            if cnt >= 0: self.refresh_stats(); self.load_list(); QMessageBox.information(self, "Success", f"Updated {cnt} items.")
            else: QMessageBox.critical(self, "Error", "Batch update failed.")

    # --- Main Logic ---
    def open_db_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select DB File", "", "SQLite DB (*.db)")
        if fname:
            self.db.connect_db(fname); self.lbl_db_name.setText(os.path.basename(fname))
            self.db.sync_missing_labels()
            self.load_labels_from_db()
            self.reset_and_load(); self.refresh_stats()

    def load_labels_from_db(self):
        def fill_list(qlist, category, defaults):
            qlist.clear()
            saved = self.db.fetch_label_settings(category)
            if not saved: saved = defaults; self.db.save_label_settings(category, saved)
            qlist.addItems(saved)
        fill_list(self.list_top, 'top', self.top_labels)
        fill_list(self.list_mid, 'mid', self.mid_labels)
        fill_list(self.list_bot, 'bot', self.bot_labels)
        self.update_combos()

    def update_combos(self):
        def fill_combo(combo, qlist):
            combo.clear(); combo.addItem("All")
            items = [qlist.item(i).text().split(' (')[0] for i in range(qlist.count())]
            combo.addItems(items)
        fill_combo(self.combo_filter_top, self.list_top)
        fill_combo(self.combo_filter_mid, self.list_mid)
        fill_combo(self.combo_filter_bot, self.list_bot)

    def save_label_order(self, category, list_widget):
        items = [list_widget.item(i).text().split(' (')[0] for i in range(list_widget.count())]
        self.db.save_label_settings(category, items)

    def load_preview(self, label_col, label_val, preview_widget):
        label_text = label_val.split(' (')[0]
        ids = self.db.get_ids_by_label(label_col, label_text)
        preview_widget.clear()
        if not ids: preview_widget.addItem("No data found.")
        else:
            for uid in ids: preview_widget.addItem(uid)

    def get_filter_opts(self):
        opts = {}
        if self.chk_date_enable.isChecked():
            opts['date_from'] = self.date_from.date().toString("yyyy-MM-dd")
            opts['date_to'] = self.date_to.date().toString("yyyy-MM-dd")
        if self.txt_serial.text().strip(): opts['serial'] = self.txt_serial.text().strip()
        if self.combo_filter_top.currentText() != "All": opts['top'] = self.combo_filter_top.currentText()
        if self.combo_filter_mid.currentText() != "All": opts['mid'] = self.combo_filter_mid.currentText()
        if self.combo_filter_bot.currentText() != "All": opts['bot'] = self.combo_filter_bot.currentText()
        opts['status'] = self.combo_status.currentText()
        return opts

    def get_logic_operator(self): return "AND" if self.rb_and.isChecked() else "OR"
    def reset_and_load(self): self.current_page = 1; self.load_list()

    def load_list(self):
        if not self.db.conn: return
        self.is_loading_table = True
        try:
            opts = self.get_filter_opts(); logic = self.get_logic_operator()
            total = self.db.get_total_count(opts, logic)
            self.total_items = total
            self.total_pages = max(1, (total + self.items_per_page - 1) // self.items_per_page)
            rows = self.db.fetch_list(self.current_page, self.items_per_page, opts, logic)
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                item_id = QTableWidgetItem(str(row['id']))
                item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, 0, item_id)
                self.table.setItem(r, 1, QTableWidgetItem(row['label_bot'] if row['label_bot'] else ""))
                self.table.setItem(r, 2, QTableWidgetItem(row['label_mid'] if row['label_mid'] else ""))
                self.table.setItem(r, 3, QTableWidgetItem(row['label_top'] if row['label_top'] else ""))
                status = "Labeled" if row['is_labeled'] else "Unlabeled"
                item = QTableWidgetItem(status); item.setForeground(QColor("#00FF00") if row['is_labeled'] else QColor("#FFaa00"))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, 4, item)
            self.lbl_page.setText(f"Page {self.current_page} / {self.total_pages} (Total {self.total_items})")
            self.btn_prev.setEnabled(self.current_page > 1); self.btn_next.setEnabled(self.current_page < self.total_pages)
        finally: self.is_loading_table = False

    def go_prev_page(self): 
        if self.current_page > 1: self.current_page -= 1; self.load_list()
    def go_next_page(self): 
        if self.current_page < self.total_pages: self.current_page += 1; self.load_list()
    
    def load_data(self, uid):
        self.current_id = uid
        raws, feat_arr, feat_names = self.db.get_signal_data(uid)
        if raws is None: return
        self.current_data = raws; self.current_feat_arr = feat_arr; self.current_feat_names = feat_names
        self.update_plots()

    def update_plots(self):
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        channels = []
        if self.chk_ch1.isChecked(): channels.append(('ch_1', 'Ch1'))
        if self.chk_ch2.isChecked(): channels.append(('ch_2', 'Ch2'))
        if self.chk_ch3.isChecked(): channels.append(('ch_3', 'Ch3'))
        if self.chk_ch4.isChecked(): channels.append(('ch_4', 'Ch4'))
        
        prev_plot = None
        cmap_name = self.combo_cmap.currentText()
        try: colormap = pg.colormap.get(cmap_name)
        except: colormap = pg.colormap.get('inferno')
        is_sync = self.chk_sync.isChecked()
        min_freq = self.spin_min.value(); max_freq = self.spin_max.value()
        
        for idx, (key, title) in enumerate(channels):
            if key not in self.current_data: continue
            sig = self.current_data[key]
            
            container = QWidget()
            cont_layout = QVBoxLayout()
            cont_layout.setContentsMargins(0,0,0,0); cont_layout.setSpacing(2)
            container.setLayout(cont_layout)
            
            title_row = QWidget(); title_layout = QHBoxLayout(); title_layout.setContentsMargins(5,0,5,0); title_row.setLayout(title_layout)
            lbl_title = QLabel(f"<b>{title}</b>")
            btn_play = QPushButton("▶ Play"); btn_play.setFixedSize(60, 24); btn_play.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 3px;")
            btn_play.clicked.connect(lambda checked, s=sig: self.play_audio(s))
            btn_stop = QPushButton("■ Stop"); btn_stop.setFixedSize(60, 24); btn_stop.setStyleSheet("background-color: #E53935; color: white; border: none; border-radius: 3px;")
            btn_stop.clicked.connect(self.stop_audio)
            title_layout.addWidget(lbl_title); title_layout.addWidget(btn_play); title_layout.addWidget(btn_stop); title_layout.addStretch()
            cont_layout.addWidget(title_row)
            
            p_widget = pg.PlotWidget(); p_widget.setMinimumHeight(200); p = p_widget.getPlotItem()
            f, t, Zxx = signal.stft(sig, fs=self.current_sr, nperseg=512, noverlap=256)
            spec_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
            p.setLabel('left', 'Freq', units='Hz'); p.setLabel('bottom', 'Time', units='s')
            p.setYRange(min_freq, max_freq, padding=0)
            
            img = pg.ImageItem(); p.addItem(img); img.setImage(spec_db.T)
            img.setRect(float(t[0]), float(f[0]), float(t[-1]-t[0]), float(f[-1]-f[0]))
            img.setLookupTable(colormap.getLookupTable())
            img.setLevels([np.max(spec_db)-80, np.max(spec_db)])
            cont_layout.addWidget(p_widget)
            if is_sync and prev_plot: p.setXLink(prev_plot)
            self.plot_layout.addWidget(container)
            prev_plot = p

    def play_audio(self, data):
        try: sd.stop(); sd.play(data, self.current_sr)
        except Exception as e: print(f"Audio Error: {e}")

    def stop_audio(self):
        try: sd.stop()
        except Exception as e: print(f"Audio Error: {e}")

    def show_feature_popup(self):
        if self.current_feat_arr is None or len(self.current_feat_arr) == 0: return
        msg = QDialog(self); msg.setWindowTitle("Features"); msg.resize(400, 500)
        txt = QTextEdit(); txt.setReadOnly(True)
        content = "Index | Name : Value\n" + "-"*35 + "\n"
        for i, val in enumerate(self.current_feat_arr):
            name = self.current_feat_names[i] if i < len(self.current_feat_names) else f"Feat_{i}"
            content += f"[{i:03d}] {name} : {val:.5f}\n"
        txt.setText(content); l = QVBoxLayout(); l.addWidget(txt); msg.setLayout(l); msg.exec_()

    def refresh_stats(self):
        if not self.db.conn: return
        stats = self.db.get_label_stats()
        def up(qlist, s_dict):
            for i in range(qlist.count()):
                it = qlist.item(i); base = it.text().split(' (')[0]
                it.setText(f"{base} ({s_dict.get(base, 0)})")
        up(self.list_top, stats.get('label_top', {}))
        up(self.list_mid, stats.get('label_mid', {}))
        up(self.list_bot, stats.get('label_bot', {}))

    def save_and_next(self):
        if not self.current_id: return
        top = self.list_top.currentItem()
        mid = self.list_mid.currentItem()
        bot = self.list_bot.currentItem()
        if not top: QMessageBox.warning(self, "Warning", "Top Label is mandatory!"); return
        
        t = top.text().split(' (')[0]
        m = mid.text().split(' (')[0] if mid else "Unlabeled"
        b = bot.text().split(' (')[0] if bot else "Unlabeled"
        
        if self.db.update_labels(self.current_id, t, m, b):
            self.refresh_stats()
            self.is_loading_table = True
            curr_row = self.table.currentRow()
            if curr_row >= 0:
                self.table.item(curr_row, 1).setText(b); self.table.item(curr_row, 2).setText(m); self.table.item(curr_row, 3).setText(t)
                status_item = self.table.item(curr_row, 4); status_item.setText("Labeled"); status_item.setForeground(QColor("#00FF00"))
            self.is_loading_table = False
            
            if curr_row < self.table.rowCount() - 1:
                self.table.selectRow(curr_row + 1); self.on_table_click(curr_row + 1, 0)
            else:
                if self.current_page < self.total_pages:
                    self.go_next_page(); self.table.selectRow(0); self.on_table_click(0, 0)
                else:
                    QMessageBox.information(self, "Done", "Last item reached.")
        else: QMessageBox.critical(self, "Error", "Failed to save label.")

    def export_data(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Export File", "", "CSV Files (*.csv)")
        if fname:
            try:
                count = self.db.export_training_data(fname)
                QMessageBox.information(self, "Success", f"Exported {count} items.\n{fname}")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    # [신규] 클러스터링 다이얼로그 열기
    def open_clustering_dialog(self):
        if not self.db.conn: return
        
        # 현재 필터 조건에 맞는 모든 ID 가져오기
        opts = self.get_filter_opts()
        logic = self.get_logic_operator()
        filtered_ids = self.db.get_filtered_ids(opts, logic)
        
        if not filtered_ids:
            QMessageBox.warning(self, "Warning", "No data found with current filters.")
            return
            
        dlg = ClusteringDialog(self, self.db, filtered_ids)
        if dlg.exec_() == QDialog.Accepted:
            # 작업 완료 후 리스트 및 통계 갱신
            self.load_labels_from_db() # 새 라벨이 생겼을 수 있으므로
            self.refresh_stats()
            self.load_list()
            
if __name__ == "__main__":
    pg.mkQApp()
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv)
    window = LabelerApp()
    window.show()
    sys.exit(app.exec_())