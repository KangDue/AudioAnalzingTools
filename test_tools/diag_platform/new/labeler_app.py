import sys
import os
import pickle

# PyQt5 강제 설정
os.environ["QT_API"] = "pyqt5"

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QKeySequence

import sounddevice as sd
import pyqtgraph as pg
from scipy import signal
import numpy as np
from backend import DataManager

# Sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# [Helper] Numpy only prediction
def extract_rf_to_dict(pipeline, target_col):
    scaler = pipeline.named_steps['scaler']
    imputer = pipeline.named_steps['imputer']
    rf = pipeline.named_steps['rf']
    
    model_data = {
        'target_col': target_col,
        'classes': rf.classes_.tolist(),
        'n_features': rf.n_features_in_,
        'imputer_statistics': imputer.statistics_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'trees': []
    }
    for estimator in rf.estimators_:
        tree = estimator.tree_
        tree_data = {
            'children_left': tree.children_left,
            'children_right': tree.children_right,
            'feature': tree.feature,
            'threshold': tree.threshold,
            'value': tree.value.reshape((-1, len(rf.classes_)))
        }
        model_data['trees'].append(tree_data)
    return model_data

def predict_rf_custom(X, model_data):
    means = model_data['imputer_statistics']
    inds = np.where(np.isnan(X))
    X[inds] = np.take(means, inds[1])
    X = (X - model_data['scaler_mean']) / model_data['scaler_scale']
    
    n_samples = X.shape[0]
    n_classes = len(model_data['classes'])
    probas = np.zeros((n_samples, n_classes))
    
    for tree in model_data['trees']:
        left = tree['children_left']
        right = tree['children_right']
        feature = tree['feature']
        threshold = tree['threshold']
        value = tree['value']
        
        node_indices = np.zeros(n_samples, dtype=np.int64)
        while True:
            is_leaf = (left[node_indices] == -1)
            if np.all(is_leaf): break
            not_leaf = ~is_leaf
            curr_nodes = node_indices[not_leaf]
            values = X[not_leaf, feature[curr_nodes]]
            go_left = values <= threshold[curr_nodes]
            node_indices[not_leaf] = np.where(go_left, left[curr_nodes], right[curr_nodes])
            
        tree_preds = value[node_indices]
        tree_preds /= tree_preds.sum(axis=1, keepdims=True)
        probas += tree_preds

    probas /= len(model_data['trees'])
    final_indices = np.argmax(probas, axis=1)
    confidence = np.max(probas, axis=1)
    
    classes = np.array(model_data['classes'])
    return classes[final_indices], confidence

class ShortcutSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shortcut Settings")
        self.resize(400, 300)
        self.settings = QSettings("FactoryLabeler", "Shortcuts")
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout(); self.setLayout(layout)
        self.table = QTableWidget(); self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Key", "Target", "Label Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        form = QHBoxLayout()
        self.txt_key = QKeySequenceEdit()
        self.combo_target = QComboBox(); self.combo_target.addItems(["label_bot", "label_mid", "label_top"])
        self.txt_val = QLineEdit(); self.txt_val.setPlaceholderText("Label Value (e.g. OK)")
        self.btn_add = QPushButton("Add"); self.btn_add.clicked.connect(self.add_shortcut)
        self.btn_del = QPushButton("Del"); self.btn_del.clicked.connect(self.del_shortcut)
        form.addWidget(QLabel("Key:")); form.addWidget(self.txt_key); form.addWidget(self.combo_target)
        form.addWidget(self.txt_val); form.addWidget(self.btn_add); form.addWidget(self.btn_del)
        layout.addLayout(form)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save_settings); btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def add_shortcut(self):
        key = self.txt_key.keySequence().toString()
        if not key or not self.txt_val.text().strip(): return
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(key))
        self.table.setItem(r, 1, QTableWidgetItem(self.combo_target.currentText()))
        self.table.setItem(r, 2, QTableWidgetItem(self.txt_val.text()))
        self.txt_key.clear(); self.txt_val.clear()

    def del_shortcut(self):
        if self.table.currentRow() >= 0: self.table.removeRow(self.table.currentRow())

    def save_settings(self):
        data = {}
        for r in range(self.table.rowCount()):
            data[self.table.item(r, 0).text()] = {'target': self.table.item(r, 1).text(), 'value': self.table.item(r, 2).text()}
        self.settings.setValue("key_map", data); self.accept()

    def load_settings(self):
        data = self.settings.value("key_map", {})
        for k, v in data.items():
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(k))
            self.table.setItem(r, 1, QTableWidgetItem(v['target']))
            self.table.setItem(r, 2, QTableWidgetItem(v['value']))

class ClusteringDialog(QDialog):
    def __init__(self, parent=None, db_manager=None, filtered_ids=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Labeling (Clustering)")
        self.resize(500, 600)
        self.db = db_manager
        self.target_ids = filtered_ids
        self.init_ui()
    def init_ui(self):
        l = QVBoxLayout(); self.setLayout(l)
        l.addWidget(QLabel(f"Target Items: {len(self.target_ids)} items"))
        algo_grp = QGroupBox("Algorithm"); l_algo = QFormLayout()
        self.combo_algo = QComboBox(); self.combo_algo.addItems(["K-Means", "DBSCAN", "Agglomerative"])
        self.spin_p1 = QSpinBox(); self.spin_p1.setRange(2, 50); self.spin_p1.setValue(5)
        self.spin_p2 = QSpinBox(); self.spin_p2.setRange(2, 50); self.spin_p2.setValue(5)
        l_algo.addRow("Algo:", self.combo_algo); l_algo.addRow("Param 1:", self.spin_p1); l_algo.addRow("Param 2:", self.spin_p2)
        algo_grp.setLayout(l_algo); l.addWidget(algo_grp)
        lbl_grp = QGroupBox("Strategy"); l_lbl = QFormLayout()
        self.combo_target = QComboBox(); self.combo_target.addItems(["label_mid", "label_bot"])
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Overwrite", "Append", "Custom"])
        self.txt_pre = QLineEdit(); l_lbl.addRow("Target:", self.combo_target); l_lbl.addRow("Mode:", self.combo_mode); l_lbl.addRow("Prefix:", self.txt_pre)
        lbl_grp.setLayout(l_lbl); l.addWidget(lbl_grp)
        self.btn_run = QPushButton("Run"); self.btn_run.clicked.connect(self.run)
        l.addWidget(self.btn_run)
    def run(self):
        if not self.target_ids: return
        self.btn_run.setEnabled(False); QApplication.processEvents()
        try:
            X, v_ids = self.db.get_features_for_clustering(self.target_ids)
            if X is None: raise Exception("No features")
            X = SimpleImputer().fit_transform(X); X = StandardScaler().fit_transform(X)
            algo = self.combo_algo.currentText()
            if algo=="K-Means": labels = KMeans(n_clusters=self.spin_p1.value(), n_init='auto').fit_predict(X)
            elif algo=="DBSCAN": labels = DBSCAN(eps=self.spin_p1.value()*0.1, min_samples=self.spin_p2.value()).fit_predict(X)
            else: labels = AgglomerativeClustering(n_clusters=self.spin_p1.value()).fit_predict(X)
            
            target = self.combo_target.currentText(); mode = self.combo_mode.currentText(); pre = self.txt_pre.text().strip()
            up_map = {}; exist = {}
            if "Append" in mode:
                self.db.cursor.execute(f"SELECT id, {target} FROM noise_data WHERE id IN ({','.join(['?']*len(v_ids))})", v_ids)
                exist = dict(self.db.cursor.fetchall())
            for uid, lid in zip(v_ids, labels):
                c_str = f"C{lid}" if lid!=-1 else "Noise"
                val = f"{pre}_{c_str}" if "Custom" in mode and pre else (f"Cluster_{c_str}" if "Overwrite" in mode else (f"{exist.get(uid,'')}+{c_str}" if exist.get(uid) else c_str))
                up_map[uid] = val
            cnt = self.db.update_labels_from_dict(up_map, target)
            cat = 'mid' if target=='label_mid' else 'bot'
            for v in set(up_map.values()): self.db.ensure_label_exists(cat, v)
            QMessageBox.information(self, "Done", f"Updated {cnt}"); self.accept()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
        finally: self.btn_run.setEnabled(True)

class TrainModelDialog(QDialog):
    def __init__(self, parent=None, db_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.resize(600, 500)
        self.db = db_manager
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 1. Settings (Target & Hyperparameters)
        set_grp = QGroupBox("Settings")
        set_layout = QFormLayout()
        
        # Target
        self.combo_target = QComboBox()
        self.combo_target.addItems(["label_mid", "label_bot", "label_top"])
        self.combo_target.currentTextChanged.connect(self.load_labels)
        
        # Toggle for Hyperparameters
        self.chk_adv = QCheckBox("Show Hyperparameters")
        self.chk_adv.toggled.connect(self.toggle_advanced)
        
        # Advanced Param Widget (Initially Hidden)
        self.adv_widget = QWidget()
        adv_layout = QFormLayout()
        self.spin_estimators = QSpinBox(); self.spin_estimators.setRange(10, 500); self.spin_estimators.setValue(100)
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(0, 100); self.spin_depth.setValue(0)
        self.spin_depth.setSpecialValueText("None (Auto)")
        
        adv_layout.addRow("n_estimators:", self.spin_estimators)
        adv_layout.addRow("max_depth:", self.spin_depth)
        self.adv_widget.setLayout(adv_layout)
        self.adv_widget.setVisible(False) # 숨김

        set_layout.addRow("Target:", self.combo_target)
        set_layout.addRow(self.chk_adv)
        set_layout.addRow(self.adv_widget)
        
        set_grp.setLayout(set_layout)
        layout.addWidget(set_grp)
        
        # 2. Label Selection
        dual_layout = QHBoxLayout()
        self.list_src = QListWidget(); self.list_src.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_dst = QListWidget(); self.list_dst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        btn_frame = QVBoxLayout()
        btn_r = QPushButton(">>"); btn_r.clicked.connect(self.mv_r)
        btn_l = QPushButton("<<"); btn_l.clicked.connect(self.mv_l)
        btn_frame.addWidget(btn_r); btn_frame.addWidget(btn_l)
        
        dual_layout.addWidget(self.list_src)
        dual_layout.addLayout(btn_frame)
        dual_layout.addWidget(self.list_dst)
        layout.addLayout(dual_layout)
        
        self.btn_run = QPushButton("Train & Save")
        self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run)
        layout.addWidget(self.btn_run)
        
        self.load_labels()

    def toggle_advanced(self, checked):
        self.adv_widget.setVisible(checked)

    def load_labels(self):
        t = self.combo_target.currentText()
        c = 'top' if 'top' in t else ('mid' if 'mid' in t else 'bot')
        self.list_src.clear(); self.list_dst.clear()
        self.list_src.addItems([x for x in self.db.fetch_label_settings(c) if x!="Unlabeled"])

    def mv_r(self): 
        for i in self.list_src.selectedItems(): self.list_dst.addItem(i.text()); self.list_src.takeItem(self.list_src.row(i))
    def mv_l(self):
        for i in self.list_dst.selectedItems(): self.list_src.addItem(i.text()); self.list_dst.takeItem(self.list_dst.row(i))

    def run(self):
        sel = [self.list_dst.item(i).text() for i in range(self.list_dst.count())]
        if not sel: QMessageBox.warning(self, "Warning", "Select labels!"); return
        
        fname, _ = QFileDialog.getSaveFileName(self, "Save", "", "Pickle (*.pkl)")
        if not fname: return
        
        self.btn_run.setEnabled(False); QApplication.processEvents()
        try:
            X, y = self.db.get_training_data(self.combo_target.currentText(), sel)
            if X is None: raise Exception("No data found.")
            
            pipe = Pipeline([
                ('imputer', SimpleImputer()), 
                ('scaler', StandardScaler()), 
                ('rf', RandomForestClassifier(
                    n_estimators=self.spin_estimators.value(), 
                    max_depth=None if self.spin_depth.value()==0 else self.spin_depth.value(), 
                    random_state=42, n_jobs=-1
                ))
            ])
            pipe.fit(X, y)
            d = extract_rf_to_dict(pipe, self.combo_target.currentText())
            with open(fname, 'wb') as f: pickle.dump(d, f, protocol=4)
            QMessageBox.information(self, "Done", "Model Saved"); self.accept()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
        finally: self.btn_run.setEnabled(True)

class PredictModelDialog(QDialog):
    def __init__(self, parent=None, db_manager=None, filtered_ids=None):
        super().__init__(parent); self.setWindowTitle("Predict"); self.db=db_manager; self.ids=filtered_ids; self.model=None; self.init_ui()
    def init_ui(self):
        l = QVBoxLayout(); self.setLayout(l)
        h = QHBoxLayout(); self.txt = QLineEdit(); btn = QPushButton("Load"); btn.clicked.connect(self.load); h.addWidget(self.txt); h.addWidget(btn); l.addLayout(h)
        self.lbl = QLabel("No Model"); l.addWidget(self.lbl)
        self.combo = QComboBox(); self.combo.addItems(["Overwrite", "Append"]); l.addWidget(self.combo)
        self.btn_run = QPushButton("Run"); self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self.run); l.addWidget(self.btn_run)
    def load(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load", "", "Pickle (*.pkl)")
        if f:
            try: 
                with open(f, 'rb') as file: d = pickle.load(file)
                self.model = d; self.txt.setText(os.path.basename(f)); self.lbl.setText(f"Target: {d['target_col']}"); self.btn_run.setEnabled(True)
            except: pass
    def run(self):
        if not self.model: return
        self.btn_run.setEnabled(False); QApplication.processEvents()
        try:
            X, v_ids = self.db.get_features_for_clustering(self.ids)
            if X is None or X.shape[1]!=self.model['n_features']: raise Exception("Mismatch")
            preds, confs = predict_rf_custom(X, self.model)
            t = self.model['target_col']; mode = self.combo.currentText()
            up_map = {}; cf_map = {}; exist = {}
            if "Append" in mode:
                self.db.cursor.execute(f"SELECT id, {t} FROM noise_data WHERE id IN ({','.join(['?']*len(v_ids))})", v_ids)
                exist = dict(self.db.cursor.fetchall())
            for uid, p, c in zip(v_ids, preds, confs):
                v = str(p)
                if "Append" in mode and exist.get(uid) and exist[uid]!="None": v = f"{exist[uid]}+{v}"
                up_map[uid] = v; cf_map[uid] = float(c)
            cnt = self.db.update_labels_from_dict(up_map, t, cf_map)
            c_key = 'top' if 'top' in t else ('mid' if 'mid' in t else 'bot')
            for v in set(up_map.values()): self.db.ensure_label_exists(c_key, v)
            QMessageBox.information(self, "Done", f"Updated {cnt}"); self.accept()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
        finally: self.btn_run.setEnabled(True)

class DataLabelerTab(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.settings = QSettings("FactoryLabeler", "GraphSettings")
        self.shortcut_settings = QSettings("FactoryLabeler", "Shortcuts")
        self.current_page = 1; self.items_per_page = 20; self.total_items = 0; self.total_pages = 1
        self.current_id = None; self.current_sr = 51200; self.current_data = {}; self.current_feat_arr = []; self.current_feat_names = []
        self.is_loading_table = False 
        self.top_labels = ["Bearing", "Others"]; self.mid_labels = ["Normal", "Noise", "Vibration"]; self.bot_labels = ["OK", "NG", "Unknown"]
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(); self.setLayout(main_layout)
        
        self.zone1 = QGroupBox("Zone 1: DB & Search"); z1_layout = QVBoxLayout()
        db_layout = QHBoxLayout()
        self.btn_open_db = QPushButton("📂 Open DB"); self.btn_open_db.clicked.connect(self.open_db_file)
        self.lbl_db_name = QLabel("No DB Loaded"); self.lbl_db_name.setStyleSheet("color: #aaa; font-size: 11px;")
        db_layout.addWidget(self.btn_open_db); db_layout.addWidget(self.lbl_db_name); z1_layout.addLayout(db_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { text-align: center; font-weight: bold; }")
        self.progress_bar.setFormat("Labeled: %v/%m (%p%)"); z1_layout.addWidget(self.progress_bar)

        search_grp = QGroupBox("Search Filters"); search_grid = QGridLayout()
        self.chk_date_enable = QCheckBox("Date Filter:"); search_grid.addWidget(self.chk_date_enable, 0, 0)
        date_layout = QHBoxLayout()
        self.date_from = QDateEdit(); self.date_from.setCalendarPopup(True); self.date_from.setDisplayFormat("yyyy-MM-dd"); self.date_from.setDate(QDate.currentDate().addMonths(-1))
        self.date_to = QDateEdit(); self.date_to.setCalendarPopup(True); self.date_to.setDisplayFormat("yyyy-MM-dd"); self.date_to.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("From")); date_layout.addWidget(self.date_from); date_layout.addWidget(QLabel("To")); date_layout.addWidget(self.date_to)
        search_grid.addLayout(date_layout, 0, 1)
        search_grid.addWidget(QLabel("Serial:"), 1, 0); self.txt_serial = QLineEdit(); search_grid.addWidget(self.txt_serial, 1, 1)
        search_grid.addWidget(QLabel("Top:"), 2, 0); self.combo_filter_top = QComboBox(); search_grid.addWidget(self.combo_filter_top, 2, 1)
        search_grid.addWidget(QLabel("Mid:"), 3, 0); self.combo_filter_mid = QComboBox(); search_grid.addWidget(self.combo_filter_mid, 3, 1)
        search_grid.addWidget(QLabel("Bot:"), 4, 0); self.combo_filter_bot = QComboBox(); search_grid.addWidget(self.combo_filter_bot, 4, 1)
        search_grid.addWidget(QLabel("Status:"), 5, 0); self.combo_status = QComboBox(); self.combo_status.addItems(["All", "Unlabeled", "Labeled"]); search_grid.addWidget(self.combo_status, 5, 1)
        logic_layout = QHBoxLayout(); self.rb_and = QRadioButton("AND"); self.rb_or = QRadioButton("OR"); self.rb_and.setChecked(True)
        logic_layout.addWidget(self.rb_and); logic_layout.addWidget(self.rb_or); search_grid.addLayout(logic_layout, 6, 0, 1, 2)
        
        btn_box1 = QHBoxLayout()
        self.btn_search = QPushButton("🔍 Filter"); self.btn_search.clicked.connect(self.reset_and_load)
        self.btn_auto = QPushButton("🤖 Cluster"); self.btn_auto.setStyleSheet("background-color: #673AB7; color: white;")
        self.btn_auto.clicked.connect(self.open_clustering_dialog)
        self.btn_del = QPushButton("🗑️ Del"); self.btn_del.setStyleSheet("background-color: #D32F2F; color: white;")
        self.btn_del.clicked.connect(self.delete_filtered_action)
        btn_box1.addWidget(self.btn_search, 2); btn_box1.addWidget(self.btn_auto, 2); btn_box1.addWidget(self.btn_del, 1)
        
        btn_box2 = QHBoxLayout()
        self.btn_train = QPushButton("🧠 Train"); self.btn_train.setStyleSheet("background-color: #00796B; color: white; font-weight: bold;")
        self.btn_train.clicked.connect(self.open_train_dialog)
        self.btn_predict = QPushButton("🔮 Predict"); self.btn_predict.setStyleSheet("background-color: #0097A7; color: white; font-weight: bold;")
        self.btn_predict.clicked.connect(self.open_predict_dialog)
        btn_box2.addWidget(self.btn_train); btn_box2.addWidget(self.btn_predict)
        
        search_grid.addLayout(btn_box1, 7, 0, 1, 2); search_grid.addLayout(btn_box2, 8, 0, 1, 2)
        search_grp.setLayout(search_grid); z1_layout.addWidget(search_grp)

        self.table = QTableWidget(); self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "Bot", "Mid", "Top", "Status", "Conf."])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self.on_table_click); self.table.currentCellChanged.connect(self.on_current_cell_changed)
        self.table.itemChanged.connect(self.on_table_item_changed); self.table.installEventFilter(self)
        z1_layout.addWidget(self.table)
        
        page_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(self.go_prev_page)
        self.lbl_page = QLabel("0 / 0"); self.lbl_page.setAlignment(Qt.AlignCenter); self.lbl_page.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(self.go_next_page)
        page_layout.addWidget(self.btn_prev); page_layout.addWidget(self.lbl_page); page_layout.addWidget(self.btn_next)
        z1_layout.addLayout(page_layout); self.zone1.setLayout(z1_layout)

        # --- Zone 2 ---
        self.zone2 = QGroupBox("Zone 2: Visualization"); z2_layout = QVBoxLayout()
        chk_style = "QCheckBox { color: black; background-color: #dddddd; padding: 4px; border-radius: 4px; font-weight: bold; }"
        ctrl_layout = QHBoxLayout()
        self.chk_ch1 = QCheckBox("Ch1"); self.chk_ch1.setChecked(True); self.chk_ch1.setStyleSheet(chk_style)
        self.chk_ch2 = QCheckBox("Ch2"); self.chk_ch2.setChecked(True); self.chk_ch2.setStyleSheet(chk_style)
        self.chk_ch3 = QCheckBox("Ch3"); self.chk_ch3.setChecked(True); self.chk_ch3.setStyleSheet(chk_style)
        self.chk_ch4 = QCheckBox("Ch4"); self.chk_ch4.setStyleSheet(chk_style)
        for c in [self.chk_ch1, self.chk_ch2, self.chk_ch3, self.chk_ch4]: c.stateChanged.connect(self.update_plots); ctrl_layout.addWidget(c)
        z2_layout.addLayout(ctrl_layout)
        
        ctrl_layout2 = QHBoxLayout()
        self.chk_sync = QCheckBox("Sync X"); self.chk_sync.setStyleSheet(chk_style); self.chk_sync.stateChanged.connect(self.update_plots)
        self.combo_cmap = QComboBox(); self.combo_cmap.addItems(['inferno', 'viridis', 'plasma', 'magma', 'jet', 'hot'])
        self.combo_cmap.currentTextChanged.connect(self.on_setting_changed)
        self.spin_min = QSpinBox(); self.spin_min.setRange(0, 25600); self.spin_min.setSingleStep(100)
        self.spin_max = QSpinBox(); self.spin_max.setRange(100, 25600); self.spin_max.setSingleStep(100)
        self.spin_min.valueChanged.connect(self.on_setting_changed); self.spin_max.valueChanged.connect(self.on_setting_changed)
        self.btn_feat = QPushButton("View Feature"); self.btn_feat.clicked.connect(self.show_feature_popup)
        ctrl_layout2.addWidget(self.chk_sync); ctrl_layout2.addWidget(QLabel("Color:")); ctrl_layout2.addWidget(self.combo_cmap)
        ctrl_layout2.addWidget(QLabel("Min:")); ctrl_layout2.addWidget(self.spin_min); ctrl_layout2.addWidget(QLabel("Max:")); ctrl_layout2.addWidget(self.spin_max)
        ctrl_layout2.addWidget(self.btn_feat); z2_layout.addLayout(ctrl_layout2)
        
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.plot_container = QWidget(); self.plot_layout = QVBoxLayout(); self.plot_layout.setSpacing(20)
        self.plot_container.setLayout(self.plot_layout); self.scroll_area.setWidget(self.plot_container)
        z2_layout.addWidget(self.scroll_area); self.zone2.setLayout(z2_layout)

        # --- Zone 3,4,5 ---
        self.zone_labels = QWidget(); label_layout = QHBoxLayout(); self.zone_labels.setLayout(label_layout)
        def create_label_zone(title, label_col, category_key):
            grp = QGroupBox(title); main_v = QVBoxLayout(); grp.setLayout(main_v)
            btn_box = QHBoxLayout()
            btn_add = QPushButton("➕"); btn_add.clicked.connect(lambda: self.add_new_label(lst, category_key))
            btn_edit = QPushButton("✎"); btn_edit.clicked.connect(lambda: self.rename_label_action(lst, label_col, category_key))
            btn_del = QPushButton("➖"); btn_del.clicked.connect(lambda: self.delete_label(lst, label_col, category_key))
            for b in [btn_add, btn_edit, btn_del]: b.setFixedWidth(40); btn_box.addWidget(b)
            main_v.addLayout(btn_box)
            
            splitter = QSplitter(Qt.Vertical)
            lst = QListWidget(); lst.setDragDropMode(QAbstractItemView.InternalMove)
            splitter.addWidget(lst) 
            prev_con = QWidget(); prev_l = QVBoxLayout(); prev_l.setContentsMargins(0,0,0,0)
            prev_l.addWidget(QLabel("▼ Comparison:")); lst_prev = QListWidget(); lst_prev.setStyleSheet("color: #ddd; background-color: #222;")
            prev_l.addWidget(lst_prev); prev_con.setLayout(prev_l); splitter.addWidget(prev_con)
            splitter.setSizes([200, 100]); main_v.addWidget(splitter)
            
            lst.model().rowsMoved.connect(lambda: self.save_label_order(category_key, lst))
            lst.itemClicked.connect(lambda item: self.load_preview(label_col, item.text(), lst_prev))
            lst_prev.itemClicked.connect(lambda item: self.load_data(item.text()))
            return grp, lst, lst_prev

        self.z3_grp, self.list_bot, self.prev_bot = create_label_zone("Zone 3: Bottom", "label_bot", "bot")
        self.z4_grp, self.list_mid, self.prev_mid = create_label_zone("Zone 4: Mid", "label_mid", "mid")
        self.z5_grp, self.list_top, self.prev_top = create_label_zone("Zone 5: Top", "label_top", "top")
        
        self.btn_save = QPushButton("💾 Set Label"); self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_and_next)
        self.btn_bulk = QPushButton("⚡ Batch Apply"); self.btn_bulk.setStyleSheet("background-color: #F57C00; color: white; padding: 8px;")
        self.btn_bulk.clicked.connect(self.bulk_update_action)
        self.btn_export = QPushButton("📤 Export"); self.btn_export.clicked.connect(self.export_data)
        
        self.z5_grp.layout().addWidget(self.btn_save); self.z5_grp.layout().addWidget(self.btn_bulk); self.z5_grp.layout().addWidget(self.btn_export)
        label_layout.addWidget(self.z3_grp); label_layout.addWidget(self.z4_grp); label_layout.addWidget(self.z5_grp)
        
        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(self.zone1); splitter.addWidget(self.zone2); splitter.addWidget(self.zone_labels)
        splitter.setSizes([400, 600, 350]); main_layout.addWidget(splitter)
        
        # Load Defaults
        self.spin_min.setValue(self.settings.value("min_freq", 0, type=int))
        self.spin_max.setValue(self.settings.value("max_freq", 25600, type=int))
        self.combo_cmap.setCurrentText(self.settings.value("colormap", "inferno", type=str))

    # --- Implement Methods ---
    def open_db_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open DB", "", "DB (*.db)")
        if f: 
            self.db.connect_db(f); self.lbl_db_name.setText(os.path.basename(f))
            self.db.sync_missing_labels(); self.load_labels_from_db(); self.reset_and_load(); self.refresh_stats()

    def load_labels_from_db(self):
        def fill(l, c, d):
            l.clear(); 
            u = QListWidgetItem("Unlabeled"); u.setForeground(QColor("#FFaa00")); l.addItem(u)
            s = self.db.fetch_label_settings(c)
            if not s: self.db.save_label_settings(c, d); s = d
            l.addItems(s)
        fill(self.list_top, 'top', self.top_labels); fill(self.list_mid, 'mid', self.mid_labels); fill(self.list_bot, 'bot', self.bot_labels)
        self.update_combos()

    def update_combos(self):
        def fill(c, l): c.clear(); c.addItem("All"); c.addItems([l.item(i).text().split(' (')[0] for i in range(l.count())])
        fill(self.combo_filter_top, self.list_top); fill(self.combo_filter_mid, self.list_mid); fill(self.combo_filter_bot, self.list_bot)

    def reset_and_load(self): self.current_page = 1; self.load_list()
    
    def load_list(self):
        if not self.db.conn: return
        self.is_loading_table = True
        try:
            opts = self.get_filter_opts(); logic = self.get_logic_operator()
            total = self.db.get_total_count(opts, logic)
            self.total_items = total
            self.total_pages = max(1, (total + self.items_per_page - 1) // self.items_per_page)
            
            labeled_cnt = self.db.get_total_count({'status': 'Labeled'}, "AND")
            total_all = self.db.get_total_count({}, "AND")
            self.progress_bar.setMaximum(total_all); self.progress_bar.setValue(labeled_cnt)
            
            rows = self.db.fetch_list(self.current_page, self.items_per_page, opts, logic)
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                # ID
                i_id = QTableWidgetItem(str(row['id'])); i_id.setFlags(i_id.flags() & ~Qt.ItemIsEditable); self.table.setItem(r, 0, i_id)
                # Labels
                self.table.setItem(r, 1, QTableWidgetItem(row['label_bot'] if row['label_bot'] else ""))
                self.table.setItem(r, 2, QTableWidgetItem(row['label_mid'] if row['label_mid'] else ""))
                self.table.setItem(r, 3, QTableWidgetItem(row['label_top'] if row['label_top'] else ""))
                # Status
                st = "Labeled" if row['is_labeled'] else "Unlabeled"
                i_st = QTableWidgetItem(st); i_st.setForeground(QColor("#00FF00") if row['is_labeled'] else QColor("#FFaa00"))
                i_st.setFlags(i_st.flags() & ~Qt.ItemIsEditable); self.table.setItem(r, 4, i_st)
                # Conf
                cf = row['confidence'] if row['confidence'] else 0.0
                i_cf = QTableWidgetItem(f"{cf:.2f}")
                if cf > 0 and cf < 0.6: i_cf.setForeground(QColor("#FF5555"))
                self.table.setItem(r, 5, i_cf)
                
            self.lbl_page.setText(f"Page {self.current_page} / {self.total_pages} (Total {self.total_items})")
            self.btn_prev.setEnabled(self.current_page > 1); self.btn_next.setEnabled(self.current_page < self.total_pages)
        finally: self.is_loading_table = False

    def load_data(self, uid):
        self.current_id = uid
        r, f, n = self.db.get_signal_data(uid)
        if r is None: return
        self.current_data = r; self.current_feat_arr = f; self.current_feat_names = n
        self.update_plots()

    def update_plots(self):
        while self.plot_layout.count(): c = self.plot_layout.takeAt(0); c.widget().deleteLater() if c.widget() else None
        chs = []
        if self.chk_ch1.isChecked(): chs.append(('ch_1', 'Ch1'))
        if self.chk_ch2.isChecked(): chs.append(('ch_2', 'Ch2'))
        if self.chk_ch3.isChecked(): chs.append(('ch_3', 'Ch3'))
        if self.chk_ch4.isChecked(): chs.append(('ch_4', 'Ch4'))
        prev = None
        try: cm = pg.colormap.get(self.combo_cmap.currentText())
        except: cm = pg.colormap.get('inferno')
        
        for k, t in chs:
            if k not in self.current_data: continue
            sig = self.current_data[k]
            con = QWidget(); l = QVBoxLayout(); l.setContentsMargins(0,0,0,0); con.setLayout(l)
            tr = QWidget(); tl = QHBoxLayout(); tl.setContentsMargins(0,0,0,0); tr.setLayout(tl)
            btn_p = QPushButton("▶"); btn_p.setFixedSize(30,20); btn_p.clicked.connect(lambda c, s=sig: self.play_audio(s))
            btn_s = QPushButton("■"); btn_s.setFixedSize(30,20); btn_s.clicked.connect(self.stop_audio)
            tl.addWidget(QLabel(f"<b>{t}</b>")); tl.addWidget(btn_p); tl.addWidget(btn_s); tl.addStretch()
            l.addWidget(tr)
            
            pw = pg.PlotWidget(); pw.setMinimumHeight(200); p = pw.getPlotItem()
            f, time, Zxx = signal.stft(sig, fs=self.current_sr, nperseg=512, noverlap=256)
            spec = 20*np.log10(np.abs(Zxx)+1e-6)
            img = pg.ImageItem(); p.addItem(img); img.setImage(spec.T)
            img.setRect(float(time[0]), float(f[0]), float(time[-1]-time[0]), float(f[-1]-f[0]))
            img.setLookupTable(cm.getLookupTable()); img.setLevels([np.max(spec)-80, np.max(spec)])
            p.setYRange(self.spin_min.value(), self.spin_max.value(), padding=0)
            if self.chk_sync.isChecked() and prev: p.setXLink(prev)
            l.addWidget(pw); self.plot_container.layout().addWidget(con); prev = p

    def play_audio(self, d): sd.stop(); sd.play(d, self.current_sr)
    def stop_audio(self): sd.stop()
    def show_feature_popup(self):
        if len(self.current_feat_arr)==0: return
        d = QDialog(self); t = QTextEdit(); t.setText(str(self.current_feat_arr)); l = QVBoxLayout(); l.addWidget(t); d.setLayout(l); d.exec_()

    def get_filter_opts(self):
        o = {}
        if self.chk_date_enable.isChecked(): o['date_from']=self.date_from.text(); o['date_to']=self.date_to.text()
        if self.txt_serial.text(): o['serial']=self.txt_serial.text()
        if self.combo_filter_top.currentText()!="All": o['top']=self.combo_filter_top.currentText()
        if self.combo_filter_mid.currentText()!="All": o['mid']=self.combo_filter_mid.currentText()
        if self.combo_filter_bot.currentText()!="All": o['bot']=self.combo_filter_bot.currentText()
        o['status']=self.combo_status.currentText()
        return o
    def get_logic_operator(self): return "AND" if self.rb_and.isChecked() else "OR"
    
    def on_current_cell_changed(self, r, c, pr, pc): 
        if not self.is_loading_table and r>=0: self.load_data(self.table.item(r, 0).text())
    def on_table_click(self, r, c): self.load_data(self.table.item(r, 0).text())
    def on_table_item_changed(self, i):
        if self.is_loading_table or i.column() in [0, 4, 5]: return
        uid = self.table.item(i.row(), 0).text(); val = i.text().strip()
        cmap = {1:('label_bot','bot'), 2:('label_mid','mid'), 3:('label_top','top')}
        if i.column() in cmap:
            col, cat = cmap[i.column()]
            self.db.update_single_label(uid, col, val)
            self.db.ensure_label_exists(cat, val)
            # Status update UI only
            has_top = self.table.item(i.row(), 3).text().strip()!=""
            self.table.item(i.row(), 4).setText("Labeled" if has_top else "Unlabeled")
            self.table.item(i.row(), 4).setForeground(QColor("#00FF00") if has_top else QColor("#FFaa00"))
            self.refresh_stats()

    def go_prev_page(self): 
        if self.current_page>1: self.current_page-=1; self.load_list()
    def go_next_page(self):
        if self.current_page<self.total_pages: self.current_page+=1; self.load_list()

    def save_label_order(self, c, l): 
        # [수정] 저장 시 숫자 떼고 이름만 저장
        items = [l.item(i).text().split(' (')[0] for i in range(l.count())]
        self.db.save_label_settings(c, items)
        
    def load_preview(self, c, v, l): 
        l.clear(); l.addItems(self.db.get_ids_by_label(c, v))
    
    def add_new_label(self, l, c):
        t, ok = QInputDialog.getText(self, "Add", "Name:"); 
        if ok and t: l.addItem(t); self.save_label_order(c, l); self.update_combos()
    def rename_label_action(self, l, col, c):
        if not l.currentItem(): return
        old = l.currentItem().text().split(' (')[0]
        new, ok = QInputDialog.getText(self, "Rename", "To:", text=old)
        if ok and new and new!=old: self.db.rename_label(c, col, old, new); self.load_labels_from_db(); self.refresh_stats(); self.load_list()
        
    # [수정] 삭제 시 UI 즉시 제거 및 DB 동기화
    def delete_label(self, l, col, c):
        if not l.currentItem(): return
        if QMessageBox.question(self, "Del", "Sure?")==QMessageBox.Yes:
            label_text = l.currentItem().text().split(' (')[0]
            if self.db.remove_label_from_records(col, label_text):
                l.takeItem(l.row(l.currentItem())) # 리스트에서 제거
                self.save_label_order(c, l)        # 변경된 리스트 DB에 저장
                self.refresh_stats(); self.load_list()

    def save_and_next(self):
        if not self.current_id: return
        t = self.list_top.currentItem().text().split(' (')[0] if self.list_top.currentItem() else ""
        if not t: QMessageBox.warning(self, "Warn", "Top Label needed"); return
        m = self.list_mid.currentItem().text().split(' (')[0] if self.list_mid.currentItem() else ""
        b = self.list_bot.currentItem().text().split(' (')[0] if self.list_bot.currentItem() else ""
        if self.db.update_labels(self.current_id, t, m, b):
            r = self.table.currentRow()
            self.table.item(r, 1).setText(b); self.table.item(r, 2).setText(m); self.table.item(r, 3).setText(t)
            self.table.item(r, 4).setText("Labeled"); self.table.item(r, 4).setForeground(QColor("#00FF00"))
            self.refresh_stats()
            if r < self.table.rowCount()-1: self.table.selectRow(r+1); self.on_table_click(r+1, 0)
            else: self.go_next_page()

    def bulk_update_action(self):
        t = self.list_top.currentItem().text().split(' (')[0] if self.list_top.currentItem() else None
        m = self.list_mid.currentItem().text().split(' (')[0] if self.list_mid.currentItem() else None
        b = self.list_bot.currentItem().text().split(' (')[0] if self.list_bot.currentItem() else None
        if not t: QMessageBox.warning(self, "Warn", "Select Top"); return
        if QMessageBox.question(self, "Batch", "Sure?")==QMessageBox.Yes:
            self.db.bulk_update_labels(self.get_filter_opts(), self.get_logic_operator(), t, m, b)
            self.refresh_stats(); self.load_list()

    def export_data(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export", "", "CSV (*.csv)")
        if f: 
            try: self.db.export_training_data(f); QMessageBox.information(self, "OK", "Done")
            except Exception as e: QMessageBox.critical(self, "Err", str(e))

    def delete_filtered_action(self):
        if QMessageBox.question(self, "Delete All", "Really delete filtered data?")==QMessageBox.Yes:
            cnt = self.db.delete_filtered_data(self.get_filter_opts(), self.get_logic_operator())
            self.refresh_stats(); self.reset_and_load(); QMessageBox.information(self, "Done", f"Deleted {cnt}")

    def open_train_dialog(self): 
        if self.db.conn: TrainModelDialog(self, self.db).exec_()
    def open_predict_dialog(self):
        if not self.db.conn: return
        ids = self.db.get_filtered_ids(self.get_filter_opts(), self.get_logic_operator())
        if ids: 
            if PredictModelDialog(self, self.db, ids).exec_()==QDialog.Accepted: self.load_list(); self.refresh_stats()
    def open_clustering_dialog(self):
        if not self.db.conn: return
        ids = self.db.get_filtered_ids(self.get_filter_opts(), self.get_logic_operator())
        if ids:
            if ClusteringDialog(self, self.db, ids).exec_()==QDialog.Accepted: self.load_labels_from_db(); self.refresh_stats(); self.load_list()
    
    def on_setting_changed(self):
        self.settings.setValue("colormap", self.combo_cmap.currentText())
        self.settings.setValue("min_freq", self.spin_min.value())
        self.settings.setValue("max_freq", self.spin_max.value())
        self.update_plots()
        
    def refresh_stats(self):
        if not self.db.conn: return
        stats = self.db.get_label_stats()
        for l, c in zip([self.list_top, self.list_mid, self.list_bot], ['label_top', 'label_mid', 'label_bot']):
            d = stats.get(c, {})
            for i in range(l.count()):
                t = l.item(i).text().split(' (')[0]
                l.item(i).setText(f"{t} ({d.get(t, 0)})")
    
    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            if self.handle_shortcut(event): return True 
            if source == self.table:
                if event.key() == Qt.Key_Down:
                    current_row = self.table.currentRow()
                    if current_row == self.table.rowCount() - 1 and self.current_page < self.total_pages:
                        self.go_next_page(); self.table.selectRow(0); self.on_table_click(0, 0); return True
                elif event.key() == Qt.Key_Up:
                    current_row = self.table.currentRow()
                    if current_row == 0 and self.current_page > 1:
                        self.go_prev_page(); last_row = self.table.rowCount() - 1
                        self.table.selectRow(last_row); self.on_table_click(last_row, 0); return True
        return super().eventFilter(source, event)

    def handle_shortcut(self, event):
        key_combine = QKeySequence(event.modifiers() | event.key()).toString()
        if not key_combine: return False
        kmap = self.shortcut_settings.value("key_map", {})
        if key_combine in kmap and self.current_id:
            act = kmap[key_combine]; target = act['target']; val = act['value']
            self.db.update_single_label(self.current_id, target, val)
            self.db.ensure_label_exists(target.replace('label_', ''), val)
            row = self.table.currentRow()
            if row >= 0:
                col_map = {'label_bot': 1, 'label_mid': 2, 'label_top': 3}
                if target in col_map:
                    self.table.item(row, col_map[target]).setText(val)
                    top_item = self.table.item(row, 3)
                    has_top = top_item.text().strip() != ""
                    st_item = self.table.item(row, 4)
                    st_item.setText("Labeled" if has_top else "Unlabeled")
                    st_item.setForeground(QColor("#00FF00") if has_top else QColor("#FFaa00"))
            
            # [핵심 수정] 순서 변경: 로드 먼저 하고 -> 통계 업데이트 (숫자 꼬임 방지)
            self.load_labels_from_db()
            self.refresh_stats()
            
            return True
        return False

class LabelerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Labeler (Pro)")
        self.resize(1400, 900)
        self.db_manager = DataManager()
        
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        act_sc = QAction("⌨ Shortcuts", self); act_sc.setShortcut("Ctrl+K")
        act_sc.triggered.connect(lambda: ShortcutSettingsDialog(self).exec_())
        file_menu.addAction(act_sc)
        
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
            QSplitter::handle { background-color: #444; }
            QSplitter::handle:hover { background-color: #0078D7; }
        """)
        
        self.labeler_tab = DataLabelerTab(self.db_manager)
        self.setCentralWidget(self.labeler_tab)

if __name__ == "__main__":
    pg.mkQApp()
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv)
    window = LabelerApp()
    window.show()
    sys.exit(app.exec_())