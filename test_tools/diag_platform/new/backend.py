import os
import sqlite3
import h5py
import orjson
import numpy as np
import pandas as pd

class DataManager:
    def __init__(self):
        self.db_path = ""
        self.h5_path = ""
        self.conn = None
        self.cursor = None

    def generate_fixed_names(self):
        channels = ['ch_1', 'ch_2', 'ch_3', 'ch_4']
        metrics = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        names = []
        for ch in channels:
            names.append(f"{ch}_SPL")
            for m in metrics:
                for b in bands:
                    names.append(f"{ch}_{m}_{b}")
        return names

    def close_db(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            self.db_path = ""
            self.h5_path = ""

    def connect_db(self, target_path):
        if os.path.isdir(target_path):
            self.db_path = os.path.join(target_path, "factory_noise.db")
        else:
            self.db_path = target_path
            
        base_name = os.path.splitext(self.db_path)[0]
        self.h5_path = f"{base_name}.h5"
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='noise_data';")
        if self.cursor.fetchone():
            self.cursor.execute("PRAGMA table_info(noise_data)")
            cols = [c[1] for c in self.cursor.fetchall()]
            if 'year' not in cols:
                self.cursor.execute("DROP TABLE noise_data"); self.conn.commit()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS noise_data (
                id TEXT PRIMARY KEY, filename TEXT, file_path TEXT,
                year INTEGER, month INTEGER, day INTEGER, serial TEXT,
                duration REAL, sample_rate INTEGER,
                label_top TEXT DEFAULT NULL, label_mid TEXT DEFAULT NULL, label_bot TEXT DEFAULT NULL,
                is_labeled INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS label_settings (
                category TEXT, name TEXT, ord INTEGER,
                PRIMARY KEY(category, name)
            )
        """)

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON noise_data (year, month, day);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_serial ON noise_data (serial);")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_label_top ON noise_data (label_top);")
        self.conn.commit()

    # [복구됨] Labeler App에서 그래프 그릴 때 사용하는 필수 함수
    def get_signal_data(self, unique_id):
        try:
            with h5py.File(self.h5_path, 'r') as h5f:
                if unique_id not in h5f: return None, None, None
                
                # Raw Data Read
                raw_grp = h5f[unique_id]['raw']
                raw_data = {k: raw_grp[k][:] for k in raw_grp.keys()}
                
                # Feature Data Read
                feat_array = np.array([])
                feat_names = []
                
                if 'feature' in h5f[unique_id]:
                    feat_array = h5f[unique_id]['feature'][:]
                    feat_names = self.generate_fixed_names()
                
                return raw_data, feat_array, feat_names
        except Exception:
            return None, None, None

    # Clustering용 대량 Feature 로드 (Global Matrix 사용)
    def get_features_for_clustering(self, id_list):
        if not id_list: return None, None
        target_ids_set = set(id_list)
        try:
            with h5py.File(self.h5_path, 'r') as h5f:
                if 'global_features' in h5f and 'global_ids' in h5f:
                    all_feats = h5f['global_features'][:]
                    all_ids = h5f['global_ids'][:].astype(str)
                    mask = np.isin(all_ids, list(target_ids_set))
                    return all_feats[mask], all_ids[mask]
                else:
                    # Fallback
                    features = []; valid_ids = []
                    for uid in id_list:
                        if uid in h5f and 'feature' in h5f[uid]:
                            features.append(h5f[uid]['feature'][:])
                            valid_ids.append(uid)
                    if not features: return None, None
                    return np.array(features), valid_ids
        except Exception as e:
            print(f"Feature Load Error: {e}")
            return None, None

    # Ingester용 배치 저장
    def insert_batch_records(self, records):
        if not records: return True
        try:
            self.conn.execute("BEGIN TRANSACTION")
            sql_data = []
            new_feats_list = []
            new_ids_list = []
            
            for meta, raw_data, feat_array in records:
                sql_data.append((
                    meta['id'], meta['filename'], meta['path'], 
                    meta['year'], meta['month'], meta['day'], meta['serial'],
                    meta['duration'], meta['sr']
                ))
                new_feats_list.append(feat_array)
                new_ids_list.append(meta['id'])

            self.cursor.executemany("""
                INSERT INTO noise_data (id, filename, file_path, year, month, day, serial, duration, sample_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sql_data)
            
            with h5py.File(self.h5_path, 'a') as h5f:
                if 'global_feat_names' not in h5f.attrs: pass 
                
                # 개별 저장
                for meta, raw_data, feat_array in records:
                    uid = meta['id']
                    if uid in h5f: del h5f[uid]
                    grp = h5f.create_group(uid)
                    raw_grp = grp.create_group('raw')
                    for ch, data in raw_data.items():
                        raw_grp.create_dataset(ch, data=data, compression="lzf")
                    grp.create_dataset('feature', data=feat_array)
                
                # Global Matrix 저장
                num_features = len(new_feats_list[0])
                if 'global_features' not in h5f:
                    h5f.create_dataset('global_features', shape=(0, num_features), maxshape=(None, num_features), dtype='f4')
                    dt_str = h5py.special_dtype(vlen=str)
                    h5f.create_dataset('global_ids', shape=(0,), maxshape=(None,), dtype=dt_str)
                
                dset_feat = h5f['global_features']
                dset_ids = h5f['global_ids']
                curr_len = dset_feat.shape[0]
                add_len = len(new_feats_list)
                
                dset_feat.resize((curr_len + add_len, num_features))
                dset_ids.resize((curr_len + add_len,))
                dset_feat[curr_len:] = np.array(new_feats_list, dtype=np.float32)
                dset_ids[curr_len:] = np.array(new_ids_list)
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Batch Insert Error: {e}")
            if self.conn: self.conn.rollback()
            return False

    def rename_label(self, category, col_name, old_name, new_name):
        try:
            self.cursor.execute("UPDATE label_settings SET name = ? WHERE category = ? AND name = ?", 
                                (new_name, category, old_name))
            query = f"UPDATE noise_data SET {col_name} = ? WHERE {col_name} = ?"
            self.cursor.execute(query, (new_name, old_name))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Rename Error: {e}")
            return False

    def bulk_update_labels(self, filters, logic, top, mid, bot):
        try:
            where_clause, params = self._build_query(filters, logic)
            set_clauses = []
            update_params = []
            
            if top:
                set_clauses.append("label_top = ?"); update_params.append(top)
            if mid and mid != "Unlabeled":
                set_clauses.append("label_mid = ?"); update_params.append(mid)
            if bot and bot != "Unlabeled":
                set_clauses.append("label_bot = ?"); update_params.append(bot)
            
            if not set_clauses: return 0

            if top: set_clauses.append("is_labeled = 1")

            query = f"UPDATE noise_data SET {', '.join(set_clauses)} {where_clause}"
            full_params = update_params + params
            
            self.cursor.execute(query, full_params)
            updated_rows = self.cursor.rowcount
            self.conn.commit()
            return updated_rows
        except Exception as e:
            print(f"Bulk Update Error: {e}")
            return -1

    def remove_label_from_records(self, col_name, label_value):
        try:
            query = f"UPDATE noise_data SET {col_name} = NULL WHERE {col_name} = ?"
            self.cursor.execute(query, (label_value,))
            self.cursor.execute("UPDATE noise_data SET is_labeled = 0 WHERE label_top IS NULL OR label_top = ''")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Delete Label Error: {e}")
            return False

    def update_labels_from_dict(self, update_map, target_col):
        if not update_map: return 0
        try:
            self.conn.execute("BEGIN TRANSACTION")
            data_to_update = [(lbl, uid) for uid, lbl in update_map.items()]
            query = f"UPDATE noise_data SET {target_col} = ? WHERE id = ?"
            self.cursor.executemany(query, data_to_update)
            
            # Status 동기화
            placeholders = ','.join(['?'] * len(update_map))
            sync_query = f"""
                UPDATE noise_data 
                SET is_labeled = CASE 
                    WHEN label_top IS NOT NULL AND label_top != '' THEN 1 
                    ELSE 0 
                END
                WHERE id IN ({placeholders})
            """
            self.cursor.execute(sync_query, list(update_map.keys()))
            
            self.conn.commit()
            return len(data_to_update)
        except Exception as e:
            print(f"Batch Dict Update Error: {e}")
            if self.conn: self.conn.rollback()
            return 0

    def update_labels(self, uid, t, m, b):
        try:
            is_labeled = 1 if (t and t.strip()) else 0
            self.cursor.execute("UPDATE noise_data SET label_top=?, label_mid=?, label_bot=?, is_labeled=? WHERE id=?", (t, m, b, is_labeled, uid))
            self.conn.commit()
            return True
        except: return False

    def update_single_label(self, uid, col_name, value):
        try:
            if not value or value.strip() == "": value = None
            query = f"UPDATE noise_data SET {col_name} = ? WHERE id = ?"
            self.cursor.execute(query, (value, uid))
            self.cursor.execute("""
                UPDATE noise_data 
                SET is_labeled = CASE 
                    WHEN label_top IS NOT NULL AND label_top != '' THEN 1 
                    ELSE 0 
                END 
                WHERE id = ?
            """, (uid,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Single Update Error: {e}")
            return False

    def fetch_label_settings(self, category):
        try:
            self.cursor.execute("SELECT name FROM label_settings WHERE category = ? ORDER BY ord ASC", (category,))
            return [r['name'] for r in self.cursor.fetchall()]
        except: return []

    def save_label_settings(self, category, label_list):
        try:
            self.cursor.execute("DELETE FROM label_settings WHERE category = ?", (category,))
            data = [(category, name, idx) for idx, name in enumerate(label_list)]
            self.cursor.executemany("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", data)
            self.conn.commit()
        except Exception as e: print(f"Save Label Config Error: {e}")

    def sync_missing_labels(self):
        categories = {'label_top': 'top', 'label_mid': 'mid', 'label_bot': 'bot'}
        for col, cat in categories.items():
            self.cursor.execute(f"SELECT DISTINCT {col} FROM noise_data WHERE {col} IS NOT NULL")
            used_labels = set([r[0] for r in self.cursor.fetchall()])
            current_labels = set(self.fetch_label_settings(cat))
            missing = used_labels - current_labels
            if missing:
                start_ord = 1000
                data = [(cat, lbl, start_ord + i) for i, lbl in enumerate(missing)]
                self.cursor.executemany("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", data)
        self.conn.commit()

    def ensure_label_exists(self, category, label_name):
        if not label_name: return False
        try:
            self.cursor.execute("SELECT 1 FROM label_settings WHERE category=? AND name=?", (category, label_name))
            if not self.cursor.fetchone():
                self.cursor.execute("SELECT MAX(ord) FROM label_settings WHERE category=?", (category,))
                res = self.cursor.fetchone()
                max_ord = res[0] if res[0] is not None else 0
                self.cursor.execute("INSERT INTO label_settings (category, name, ord) VALUES (?, ?, ?)", (category, label_name, max_ord + 1))
                self.conn.commit()
                return True
            return False
        except: return False

    def insert_record(self, meta, raw_data, feat_array):
        # 단건도 Matrix 호환성을 위해 Batch 함수 사용 권장
        return self.insert_batch_records([(meta, raw_data, feat_array)])

    def get_all_existing_ids(self):
        if not self.cursor: return set()
        self.cursor.execute("SELECT id FROM noise_data")
        return set(row[0] for row in self.cursor.fetchall())

    def get_filtered_ids(self, filters, logic="AND"):
        w, p = self._build_query(filters, logic)
        self.cursor.execute(f"SELECT id FROM noise_data {w}", p)
        return [r[0] for r in self.cursor.fetchall()]

    def export_training_data(self, output_path):
        self.cursor.execute("SELECT id, label_bot, label_mid, label_top FROM noise_data WHERE is_labeled=1")
        rows = self.cursor.fetchall()
        if not rows: raise Exception("라벨링된 데이터가 없습니다.")
        fixed_names = self.generate_fixed_names()
        export_data = []
        if os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, 'r') as h5f:
                for row in rows:
                    uid = row['id']
                    if uid not in h5f: continue
                    # Feature 개별 저장된 것 읽기
                    if 'feature' not in h5f[uid]: continue
                    feats = h5f[uid]['feature'][:]
                    if len(feats) != len(fixed_names): continue
                    row_dict = {"file_id": uid}
                    for n, v in zip(fixed_names, feats): row_dict[n] = float(v)
                    row_dict["label_bot"] = row['label_bot']
                    row_dict["label_mid"] = row['label_mid']
                    row_dict["label_top"] = row['label_top']
                    export_data.append(row_dict)
        if export_data:
            df = pd.DataFrame(export_data)
            final_cols = ['file_id'] + fixed_names + ['label_bot', 'label_mid', 'label_top']
            final_cols = [c for c in final_cols if c in df.columns]
            df = df[final_cols]
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            return len(df)
        else: raise Exception("Export할 데이터가 없습니다.")

    def check_id_exists(self, unique_id):
        if not self.cursor: return False
        self.cursor.execute("SELECT 1 FROM noise_data WHERE id = ?", (unique_id,))
        return self.cursor.fetchone() is not None

    def _build_query(self, filters, logic="AND"):
        conditions = []; params = []
        if filters.get('date_from') and filters.get('date_to'):
            try:
                start_date = int(filters['date_from'].replace('-', ''))
                end_date = int(filters['date_to'].replace('-', ''))
                conditions.append("(year * 10000 + month * 100 + day) BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            except ValueError: pass
        if filters.get('serial'): conditions.append("serial LIKE ?"); params.append(f"%{filters['serial']}%")
        if filters.get('top'): conditions.append("label_top = ?"); params.append(filters['top'])
        if filters.get('mid'): conditions.append("label_mid = ?"); params.append(filters['mid'])
        if filters.get('bot'): conditions.append("label_bot = ?"); params.append(filters['bot'])
        base = f"WHERE ({f' {logic} '.join(conditions)})" if conditions else ""
        if filters.get('status') == 'Labeled': base += (" AND" if base else " WHERE") + " is_labeled = 1"
        elif filters.get('status') == 'Unlabeled': base += (" AND" if base else " WHERE") + " is_labeled = 0"
        return base, params

    def get_total_count(self, filters, logic="AND"):
        w, p = self._build_query(filters, logic)
        self.cursor.execute(f"SELECT COUNT(*) FROM noise_data {w}", p)
        return self.cursor.fetchone()[0]

    def fetch_list(self, page=1, per_page=20, filters=None, logic="AND"):
        w, p = self._build_query(filters, logic)
        offset = (page-1)*per_page
        p.extend([per_page, offset])
        self.cursor.execute(f"SELECT id, filename, is_labeled, label_top, label_mid, label_bot FROM noise_data {w} ORDER BY created_at DESC LIMIT ? OFFSET ?", p)
        return self.cursor.fetchall()

    def get_ids_by_label(self, col, val, limit=50):
        self.cursor.execute(f"SELECT id FROM noise_data WHERE {col} = ? LIMIT ?", (val, limit))
        return [r[0] for r in self.cursor.fetchall()]

    def get_label_stats(self):
        s = {}
        if not self.cursor: return s
        for c in ['label_top', 'label_mid', 'label_bot']:
            try:
                self.cursor.execute(f"SELECT {c}, COUNT(*) FROM noise_data WHERE {c} IS NOT NULL GROUP BY {c}")
                s[c] = dict(self.cursor.fetchall())
            except: s[c] = {}
        return s