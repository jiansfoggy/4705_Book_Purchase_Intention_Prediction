import argparse
import joblib
import os
import random
import re
import subprocess
import sys
import wandb
import warnings
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from funksvd import FunkSVD
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ndcg_score


# reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')


# ==============
# = Git Commit =
# ==============
def get_git_commit_hash():
    # Capture Code Version from Git Commit
    # Return short git commit hash
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        commit_hash = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        commit_hash = commit_hash.decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown"


def init_wandb(project_name="Book_Purchase_Prediction",
               experiment_name=None, entity=None, config=None,
               save_code=True):

    if entity is None:
        entity = os.environ.get("WANDB_ENTITY", None)

    # Initialize a new W&B run
    run = wandb.init(
        project=project_name, name=experiment_name,
        entity=entity, config=config, reinit=True)

    if save_code:
        try:
            # log_code will capture python files in repo as an artifact
            run.log_code(".")
        except Exception as e:
            # best-effort: continue even if code snapshot fails
            run.log({"_code_logging_error": str(e)})
    return run


# =================
# = Data Load and =
# = Preprocess    =
# =================
def load_and_prep(meta_path, reviews_path):
    meta = pd.read_csv(meta_path)
    reviews = pd.read_csv(reviews_path)

    # ensure parent_asin
    if 'parent_asin' not in meta.columns:
        meta['parent_asin'] = meta.get('asin', '').astype(str)

    for col in ['title', 'description', 'features', 'categories',
                'bought_together', 'average_rating', 'rating_number']:
        if col not in meta.columns:
            meta[col] = ''

    meta['title'] = meta['title'].fillna('')
    meta['description'] = meta['description'].fillna('')
    meta['features'] = meta['features'].fillna('')
    meta['categories'] = meta['categories'].fillna('')
    meta['bought_together'] = meta['bought_together'].fillna('')

    def mktext(r):
        parts = [r['title'], str(r.get('features','')),
                 str(r.get('description','')), str(r.get('categories',''))]
        return " ".join([p for p in parts if isinstance(p, str) and p.strip() != ''])
    meta = meta.drop_duplicates(subset=['parent_asin']).reset_index(drop=True)
    meta['item_text'] = meta.apply(mktext, axis=1)

    meta['rating_number'] = pd.to_numeric(meta['rating_number'], errors='coerce').fillna(0)
    if meta['rating_number'].sum() == 0 and 'parent_asin' in reviews.columns:
        counts = reviews['parent_asin'].astype(str).value_counts()
        meta['rating_number'] = meta['parent_asin'].map(counts).fillna(0).astype(int)
    meta['pop_score'] = np.log1p(meta['rating_number'])

    def parse_bought(x):
        try:
            if not x or pd.isna(x):
                return []
            return re.findall(r"[A-Z0-9]{4,}", str(x))
        except:
            return []
    meta['bought_list'] = meta['bought_together'].apply(parse_bought)
    meta['cat_list'] = meta['categories'].apply(lambda x: [t.strip().lower()
                                                           for t in re.split(r'[>|,/;]+', str(x))
                                                           if t.strip()!=''])

    item_ids = meta['parent_asin'].astype(str).tolist()
    id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    idx_to_id = {idx: iid for iid, idx in id_to_idx.items()}
    return meta, reviews, id_to_idx, idx_to_id


# ====================
# = Model Helper:    =
# = Build embeddings = 
# = (TF-IDF + SVD)   =
# ====================
def build_item_embeddings(meta, max_features=5000, svd_dim=50):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), 
                          stop_words='english')
    X = vec.fit_transform(meta['item_text'].fillna(''))
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_SEED)
    emb = svd.fit_transform(X)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return vec, svd, emb, X


# ===================
# = Model Helper:   =
# = Simple lookup   =
# = (title -> asin) =
# ===================
def lookup_asin_by_title(meta, title):
    if not isinstance(title, str) or title.strip() == '':
        return None
    t = title.strip().lower()
    df = meta[meta['title'].str.lower() == t]
    if not df.empty:
        return df['parent_asin'].iloc[0]
    df2 = meta[meta['title'].str.lower().str.contains(re.escape(t))]
    if not df2.empty:
        return df2['parent_asin'].iloc[0]
    words = [w for w in re.split(r'\W+', t) if len(w) > 2]
    for w in words:
        df3 = meta[meta['title'].str.lower().str.contains(re.escape(w))]
        if not df3.empty:
            return df3['parent_asin'].iloc[0]
    return None


# ==============
# = Evaluation =
# ==============
def build_user_items(df):
    d = {}
    for _, r in df.iterrows():
        uid = str(r.get('user_id','')).strip()
        asin = str(r.get('parent_asin', r.get('asin',''))).strip()
        if uid=='' or asin=='':
            continue
        d.setdefault(uid, set()).add(asin)
    return d


# evaluate_recall_at_k_using_train
def evaluate_recall_at_k(meta, id_to_idx, item_emb, train_reviews, test_reviews,
                         recommend_fn, K=10, num_eval=500, seed_per_user=1, debug=False):
    """
    recommend_fn signature now should accept user_id keyword arg:
      recommend_fn(meta, id_to_idx, item_emb, favorite_titles, topk=K, user_id=None)
    """
    train_user_items = build_user_items(train_reviews)
    test_user_items = build_user_items(test_reviews)

    items_in_meta = set(id_to_idx.keys())

    # eligible users: require seeds in meta and at least one test item in meta
    eligible = []
    for u in train_user_items.keys():
        train_items = set(train_user_items.get(u, [])) & items_in_meta
        test_items = set(test_user_items.get(u, [])) & items_in_meta
        if len(train_items) >= seed_per_user and len(test_items) >= 1:
            eligible.append(u)

    if debug:
        print(f"[eval] eligible users after filtering: {len(eligible)}")
    if len(eligible) == 0:
        return {'recall': 0.0, 'precision': 0.0, 'ndcg': 0.0}

    sampled = random.sample(eligible, min(num_eval, len(eligible)))
    hits = 0; total_rel = 0; precisions=[]; ndcgs=[]

    for u in sampled:
        train_items = list(set(train_user_items.get(u, [])) & items_in_meta)
        test_items = set(test_user_items.get(u, [])) & items_in_meta
        if not test_items or len(train_items) < seed_per_user:
            continue
        seeds = random.sample(train_items, seed_per_user)
        seed_titles = []
        for s in seeds:
            row = meta[meta['parent_asin'] == s]
            if not row.empty:
                seed_titles.append(row['title'].iloc[0])
            else:
                seed_titles.append(s)

        # **pass user id into recommend_fn**
        recs = recommend_fn(meta, id_to_idx, item_emb, seed_titles, topk=K, user_id=u)
        if recs is None or recs.empty:
            continue
        rec_set = set(recs['parent_asin'].astype(str).values)
        hits += len(rec_set.intersection(test_items))
        total_rel += len(test_items)
        precisions.append(len(rec_set.intersection(test_items))/float(K))
        rel = np.asarray([[1 if r in test_items else 0 for r in recs['parent_asin'].astype(str).values]])
        try:
            ndcgs.append(ndcg_score(rel, np.arange(K,0,-1).reshape(1,-1)))
        except:
            pass

    recall = hits/(total_rel+1e-9) if total_rel>0 else 0.0
    precision = np.mean(precisions) if precisions else 0.0
    ndcg = np.mean(ndcgs) if ndcgs else 0.0

    if debug:
        print(f"[eval] sampled {len(sampled)} users -> hits {hits}, total_rel {total_rel}, recall {recall:.6f}")

    return {'recall': recall, 'precision': precision, 'ndcg': ndcg}


# ==========================
# = Log Artifacts for Data =
# = & model in WandB       =
# ==========================
# log_data_and_model_artifacts
def create_artifacts(run, data_paths, model_path, dataset_name, model_name, alias="v1"):
    # log data artifact(s)
    data_art = wandb.Artifact(name=f"{dataset_name}-artifact", type="dataset", metadata={})
    for p in data_paths:
        if os.path.exists(p):
            data_art.add_file(p)
    run.log_artifact(data_art)
    data_art.wait()
    data_art.aliases.append("staging")

    # log model artifact
    model_art = wandb.Artifact(name=f"{model_name}-artifact", type="model", metadata={})
    model_art.add_file(model_path)
    run.log_artifact(model_art)
    # register model - some W&B APIs differ; we follow run.link_model style per example
    run.link_model(path=model_path, registered_model_name=f"{model_name}-artifact",
                   aliases=[alias])

    # Promote to Staging or Production
    model_art.wait()
    model_art.aliases.append("staging")
    print("Model registered and promoted to 'staging'.")
    return data_art, model_art


# =====================
# = Inference helpers =
# =====================
def _ensure_item_emb_from_artifact(artifact):
    if 'item_emb' in artifact and artifact['item_emb'] is not None:
        return artifact['item_emb']
    # try common names
    if 'content_obj' in artifact and isinstance(artifact['content_obj'], dict):
        obj = artifact['content_obj']
    else:
        obj = artifact
    vec = obj.get('vec') or obj.get('vectorizer') or None
    svd = obj.get('svd') or None
    meta = artifact.get('meta') or artifact.get('meta_df') or None
    if vec is not None and svd is not None and meta is not None:
        try:
            X = vec.transform(meta['item_text'].fillna(''))
            emb = svd.transform(X)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            return emb
        except Exception:
            return None
    return None


def recommend_mf_wrapper(mf_model, user_to_index_map, meta, id_to_idx, item_emb):
    def recommend_mf(meta_local, id_to_idx_local, emb_local, 
    	             favorite_titles, topk=10, user_id=None):
        uidx = user_to_index_map[user_id]
        preds = mf_model.predict_user(uidx)  # should return array of length n_items
        # exclude favorites (by asin)
        fav_asins = [lookup_asin_by_title(meta_local, t) for t in favorite_titles]
        for a in fav_asins:
            if a in id_to_idx_local:
                preds[id_to_idx_local[a]] = -1
        top_idx = np.argsort(preds)[::-1][:topk]
        cols = ['parent_asin', 'title', 'average_rating', 'rating_number']
        df = meta_local.iloc[top_idx][cols].copy()
        df['score'] = preds[top_idx]
        return df.reset_index(drop=True)
    return recommend_mf


# =============
# = Main Func =
# =============
def main(meta_csv='meta_data.csv', reviews_csv='review_data.csv',
         train_size=100000, test_size=10000, project="Personalized-Book-Recommender", 
         entity=None, max_features=5000, svd_dim=50, mf_factors=40, mf_epochs=120):
    # load data
    meta, reviews, id_to_idx, idx_to_id = load_and_prep(meta_csv, reviews_csv)
    vec, svd_model, item_emb, Xtf = build_item_embeddings(meta, max_features=max_features,
                                                          svd_dim=svd_dim)

    # sample/split reviews (random)
    total_needed = min(len(reviews), train_size + test_size)
    sampled = reviews.sample(n=total_needed, random_state=RANDOM_SEED).reset_index(drop=True)
    train_reviews = sampled.iloc[:min(train_size, len(sampled))].reset_index(drop=True)
    test_reviews = sampled.iloc[min(train_size, len(sampled)):].reset_index(drop=True)

    git_hash = get_git_commit_hash()
    dataset_version = "v1"  # you can compute hash of data if desired

    # model loop: train/eval each model, log to W&B
    results = {}
    model_name = "FunkSVD"
    exp_name = f"{model_name}-exp"
    config = {
        "git_commit": git_hash,
        "dataset": "Amazon Books subset",
        "dataset_version": dataset_version,
        "model_name": model_name,
        "max_features": max_features,
        "svd_dim": svd_dim,
        "mf_factors": mf_factors,
        "mf_epochs": mf_epochs
    }
    run = init_wandb(project_name=project, experiment_name=exp_name,
                     entity=entity, config=config, save_code=True)
    #run.config.update({"model_name": model_name})
    local_model_path = f"{model_name}_model.pkl"

    # Train FunkSVD on train_reviews
    users = train_reviews['user_id'].astype(str).unique().tolist()
    user_to_index_map_local = {u: idx for idx, u in enumerate(users)}
    user_idx = []
    item_idx = []
    ratings = []
    for _, r in train_reviews.iterrows():
        u = str(r.get('user_id',''))
        asin = str(r.get('parent_asin', r.get('asin','')))
        if u in user_to_index_map_local and asin in id_to_idx:
            user_idx.append(user_to_index_map_local[u])
            item_idx.append(id_to_idx[asin])
            ratings.append(float(r.get('rating',0.0)))
    user_idx = np.array(user_idx, dtype=np.int32)
    item_idx = np.array(item_idx, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.float32)
    mf_model = FunkSVD(n_users=len(user_to_index_map_local), n_items=len(meta), n_factors=mf_factors,
                           lr=0.01, reg=0.02, n_epochs=mf_epochs, verbose=True)
    if len(ratings) > 0:
        mf_model.fit(user_idx, item_idx, ratings)
        # compute RMSE on a subset of test pairs where both user & item known
        preds = []
        trues = []
        for _, r in test_reviews.iterrows():
            u = str(r.get('user_id',''))
            asin = str(r.get('parent_asin', r.get('asin','')))
            if u in user_to_index_map_local and asin in id_to_idx:
                ui = user_to_index_map_local[u]; ii = id_to_idx[asin]
                preds.append(mf_model.predict_pair(ui, ii))
                trues.append(float(r.get('rating',0.0)))
        if len(preds) > 0:
            rmse = float(np.sqrt(mean_squared_error(trues, preds)))
        else:
            rmse = None
    else:
        rmse = None

    mf_recommender_fn = recommend_mf_wrapper(mf_model, user_to_index_map_local,
    	                                     meta, id_to_idx, item_emb)
    metrics = evaluate_recall_at_k(meta, id_to_idx, item_emb, train_reviews,
                                   test_reviews, mf_recommender_fn, K=10, num_eval=200)

    # log metrics
    logd = {"recall@10": metrics['recall'],
            "precision@10": metrics['precision'],
            "ndcg@10": metrics['ndcg']}
    if rmse is not None:
        logd["rmse_test_pred"] = rmse
    run.log(logd)
    # save mf model + user mapping
    joblib.dump({"mf_model": mf_model, "user_to_index_map": user_to_index_map_local}, local_model_path)
    data_art, model_art = create_artifacts(run, [meta_csv, reviews_csv], local_model_path,
                                           dataset_name="books_subset", model_name=model_name, alias="production")
    results[model_name] = {"metrics": metrics, "rmse": rmse, "artifact": model_art}

    # Summarize run
    run.summary["git_commit"] = git_hash
    run.summary["dataset_version"] = dataset_version

    # After training all models: choose best by recall@10 and promote
    best_model = None
    best_recall = -1.0
    for mname, info in results.items():
        rec = info.get("metrics", {}).get("recall", 0.0)
        if rec is None:
            rec = 0.0
        if rec > best_recall:
            best_recall = rec
            best_model = mname
    print("Best model by recall@10:", best_model, best_recall)

    # Promote best model artifact to "production" alias
    if best_model and results.get(best_model):
        run.link_model(path=local_model_path, registered_model_name=f"{model_name}-artifact", aliases=["production"])
        print("Promoted best model to production")
    run.finish()
    print(f"Finished run for {model_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="../data/meta_data.csv")
    parser.add_argument("--reviews", default="../data/review_data.csv")
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--project", default="Personalized-Book-Recommender")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    # login if WANDB_API_KEY env not set
    if os.environ.get("WANDB_API_KEY", None) is None:
        print("No WANDB_API_KEY env var found.\
               You may be prompted to login via browser.")
    # run (user should be logged in)
    main(meta_csv=args.meta, reviews_csv=args.reviews, train_size=args.train_size,
         test_size=args.test_size, project=args.project, entity=args.entity)

Column names:
['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']

Column names:
['main_category', 'title', 'subtitle', 'author', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'videos', 'store', 'categories', 'details', 'parent_asin', 'bought_together']


# python3 recommand_model.py --meta ../data/meta_data.csv --reviews ../data/review_data.csv --train_size 200000 --test_size 20000 --entity jsfoggy
# python3 recommand_model.py --meta ../data/meta_data.csv --reviews ../data/review_data.csv --train_size 53000 --test_size 2000 --entity jsfoggy
