"""Find datasets: a curated offline catalog plus live OpenML and Hugging Face search."""
from __future__ import annotations

import re

import requests

# name -> (loader snippet, description). Offline ones ship with scikit-learn.
CATALOG = {
    "iris": ("sklearn.datasets.load_iris()", "150 flowers, 4 features, 3 classes. Offline."),
    "wine": ("sklearn.datasets.load_wine()", "178 wines, 13 features, 3 classes. Offline."),
    "breast_cancer": ("sklearn.datasets.load_breast_cancer()", "Wisconsin diagnostic, 569 x 30, binary. Offline."),
    "digits": ("sklearn.datasets.load_digits()", "8x8 handwritten digits, 1797 samples, 10 classes. Offline."),
    "diabetes": ("sklearn.datasets.load_diabetes()", "442 patients, 10 features, regression. Offline."),
    "california_housing": ("sklearn.datasets.fetch_california_housing()", "20640 x 8, regression, median house value (x100k). Downloads once."),
    "covertype": ("sklearn.datasets.fetch_covtype()", "581k x 54, 7 forest cover classes. Downloads once."),
    "20newsgroups": ("sklearn.datasets.fetch_20newsgroups(subset='all')", "~18k newsgroup posts, 20 classes, text. Downloads once."),
    "olivetti_faces": ("sklearn.datasets.fetch_olivetti_faces()", "400 64x64 faces, 40 people. Downloads once."),
    "kddcup99": ("sklearn.datasets.fetch_kddcup99()", "Network intrusion detection. Downloads once."),
    "mnist": ("sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False)", "70k 28x28 digits via OpenML (~50MB). Standard split: first 60k train, last 10k test."),
    "fashion_mnist": ("sklearn.datasets.fetch_openml('Fashion-MNIST', version=1, as_frame=False)", "70k 28x28 clothing images via OpenML."),
    "adult": ("sklearn.datasets.fetch_openml('adult', version=2, as_frame=True)", "Census income >50K, 48842 rows, mixed types."),
    "higgs": ("sklearn.datasets.fetch_openml('higgs', version=1)", "OpenML 98k subset of HIGGS, binary."),
    "credit_g": ("sklearn.datasets.fetch_openml('credit-g', version=1, as_frame=True)", "German credit, 1000 rows, binary."),
    "titanic": ("sklearn.datasets.fetch_openml('titanic', version=1, as_frame=True)", "1309 passengers, survival."),
    "spambase": ("sklearn.datasets.fetch_openml('spambase', version=1)", "4601 emails, 57 features, binary."),
    "bank_marketing": ("sklearn.datasets.fetch_openml('bank-marketing', version=1, as_frame=True)", "45k rows, term deposit subscription."),
    "credit_card_fraud": ("sklearn.datasets.fetch_openml('creditcard', version=1, as_frame=True)", "284k transactions, 0.17% fraud, highly imbalanced."),
    "cifar10": ("torchvision.datasets.CIFAR10(root='data', download=True)", "60k 32x32 color images, 10 classes. Needs torchvision; CPU training is slow."),
    "imdb": ("datasets.load_dataset('imdb')", "50k movie reviews, binary sentiment. Needs `datasets` package."),
    "sst2": ("datasets.load_dataset('glue', 'sst2')", "Stanford sentiment, binary. Needs `datasets` package."),
}

ALIASES = {
    "cancer": "breast_cancer", "wdbc": "breast_cancer", "housing": "california_housing",
    "census": "adult", "income": "adult", "german": "credit_g", "fraud": "credit_card_fraud",
    "newsgroups": "20newsgroups", "covtype": "covertype", "spam": "spambase", "faces": "olivetti_faces",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def search_catalog(query: str) -> list[str]:
    q = _norm(query)
    tokens = set(q.split("_"))
    found = []
    for name, (loader, desc) in CATALOG.items():
        if name in q or q in name or tokens & set(name.split("_")) or any(
            a in tokens and t == name for a, t in ALIASES.items()
        ) or any(t in _norm(desc).split("_") for t in tokens if len(t) > 3):
            found.append(f"[builtin] {name}: {desc}\n    load: {loader}")
    return found


def search_openml(query: str, limit: int = 6) -> list[str]:
    """OpenML lists by exact name, so try the query and a few variants."""
    out, seen = [], set()
    variants = {query.strip(), query.strip().lower(), query.strip().replace(" ", "-"), query.strip().replace(" ", "_")}
    for v in variants:
        try:
            r = requests.get(f"https://www.openml.org/api/v1/json/data/list/data_name/{v}/limit/{limit}", timeout=15)
        except requests.RequestException as e:
            return [f"[openml] unreachable: {e}"]
        if r.status_code != 200:
            continue
        for d in r.json().get("data", {}).get("dataset", []):
            key = (d.get("name"), d.get("version"))
            if key in seen:
                continue
            seen.add(key)
            q = {x["name"]: x["value"] for x in d.get("quality", [])}
            out.append(
                f"[openml] {d.get('name')} v{d.get('version')} (id {d.get('did')}): "
                f"{q.get('NumberOfInstances', '?')} rows, {q.get('NumberOfFeatures', '?')} features, "
                f"{q.get('NumberOfClasses', '?')} classes\n    load: sklearn.datasets.fetch_openml(data_id={d.get('did')})"
            )
    return out[:limit]


def search_huggingface(query: str, limit: int = 6) -> list[str]:
    try:
        r = requests.get("https://huggingface.co/api/datasets", params={"search": query, "limit": limit, "sort": "downloads"}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        return [f"[huggingface] unreachable: {e}"]
    return [
        f"[huggingface] {d['id']} ({d.get('downloads', 0):,} downloads)\n    load: datasets.load_dataset('{d['id']}')"
        for d in r.json()
    ]


def search_datasets(query: str, online: bool = True) -> str:
    results = search_catalog(query)
    if online:
        results += search_openml(query)
        results += search_huggingface(query)
    if not results:
        return f"No datasets found for '{query}'. Try a shorter or more common name."
    return "\n".join(results)
