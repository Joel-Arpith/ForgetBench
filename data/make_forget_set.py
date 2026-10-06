"""TOFU-style synthetic author-profile dataset: each author is a fully
made-up person, so 'should this be forgotten' has unambiguous ground truth.
No scraping, no real people.
"""
import json
import random
from pathlib import Path

FIRST = ["Elena", "Marcus", "Priya", "Tobias", "Ines", "Kwame", "Sana", "Leif",
         "Mireille", "Omar", "Yuki", "Bram", "Talia", "Costas", "Nadia"]
LAST = ["Voss", "Adeyemi", "Kowalski", "Reyes", "Lindqvist", "Haddad",
        "Okafor", "Ferreira", "Bianchi", "Nakamura", "Novak", "Duval"]
GENRE = ["speculative fiction", "historical crime", "literary memoir",
         "epic fantasy", "hard science fiction", "domestic noir"]
CITY = ["Porto", "Nairobi", "Kaunas", "Cusco", "Kyoto", "Tallinn", "Accra",
        "Ljubljana", "Recife"]

FACTS = [
    "was born in {city} in {year}.",
    "writes primarily in the {genre} genre.",
    "won the fictional Halden Prize in {year2}.",
    "published a debut novel titled '{title}'.",
    "has a sibling who works as a {job}.",
]
JOBS = ["cartographer", "marine biologist", "glassblower", "notary", "chef"]


def _author(rng, idx):
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    ctx = dict(
        city=rng.choice(CITY), year=rng.randint(1950, 1995),
        year2=rng.randint(1996, 2023), genre=rng.choice(GENRE),
        title=f"The {rng.choice(['Silent', 'Last', 'Hollow', 'Amber'])} "
              f"{rng.choice(['Garden', 'Ledger', 'Harbor', 'Choir'])}",
        job=rng.choice(JOBS),
    )
    qas = []
    for fact in FACTS:
        sentence = fact.format(**ctx)
        question = f"What is a notable fact about the author {name}?"
        answer = f"{name} {sentence}"
        qas.append({"author_id": idx, "author": name, "question": question, "answer": answer})
    return qas


def generate(n_authors: int, forget_frac: float, seed: int, out_dir: str):
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_qas = [qa for i in range(n_authors) for qa in _author(rng, i)]
    author_ids = list(range(n_authors))
    rng.shuffle(author_ids)
    n_forget = max(1, int(n_authors * forget_frac))
    forget_ids = set(author_ids[:n_forget])

    forget_set = [qa for qa in all_qas if qa["author_id"] in forget_ids]
    retain_set = [qa for qa in all_qas if qa["author_id"] not in forget_ids]

    for name, rows in [("forget.jsonl", forget_set), ("retain.jsonl", retain_set)]:
        with open(out / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return {"forget_authors": len(forget_ids), "retain_authors": n_authors - len(forget_ids),
            "forget_qas": len(forget_set), "retain_qas": len(retain_set)}


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    stats = generate(cfg.n_authors, cfg.forget_frac, cfg.seed, cfg.data_dir)
    print(stats)
