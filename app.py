import streamlit as st
import random, math, io, torch, numpy as np, re, nltk
from nltk.corpus import wordnet
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Humanizer — Everyday Language", layout="wide")
FREE_TIER_LIMIT = 10
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# -----------------------
# Load vocab
# -----------------------
@st.cache_resource
def load_common_words():
    with open("common_words.txt", "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())
COMMON_WORDS_SET = load_common_words()

# -----------------------
# Load models ONCE
# -----------------------
@st.cache_resource
def load_models():
    # 
    paraphraser = pipeline(
    "text2text-generation",
    model="Vamsi/T5_Paraphrase_Paws",
    device=0 if torch.cuda.is_available() else -1,
    tokenizer_kwargs={"use_fast": False}   # 👈 force slow tokenizer
)

    gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")  # smaller GPT2
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.eval()
    if torch.cuda.is_available():
        gpt2_model.to("cuda")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return paraphraser, gpt2_model, gpt2_tokenizer, embedder

paraphraser, gpt2_model, gpt2_tokenizer, embedder = load_models()

# -----------------------
# Helpers
# -----------------------
def simplify_to_common(text, extra_noise=False):
    words = text.split()
    simplified = []
    for w in words:
        base = w.strip(",.?!;:").lower()
        if base in COMMON_WORDS_SET or not base.isalpha():
            replacement = w
        else:
            syns = wordnet.synsets(base)
            found = None
            for syn in syns:
                for lemma in syn.lemmas():
                    cand = lemma.name().replace("_", " ").lower()
                    if cand in COMMON_WORDS_SET:
                        found = cand
                        break
                if found:
                    break
            replacement = found if found else w

        # add human imperfection noise
        if extra_noise and random.random() < 0.15:
            replacement = replacement + random.choice([" basically", " sort of", " in simple terms", " kind of"]) if random.random() < 0.3 else replacement
        simplified.append(replacement)
    return " ".join(simplified)

def clean_text_output(text):
    text = text.replace("-", " ")
    text = re.sub(r"\.\s*\.", ".", text)           
    text = re.sub(r"([!?]){2,}", r"\1", text)      
    text = re.sub(r",\s*,+", ", ", text)           
    text = re.sub(r"\s+", " ", text)               
    text = re.sub(r"\b[Pp]araphrase\b[:]*", "", text)
    text = re.sub(r"\b[Ff]alse\b[.:]*", "", text)
    return text.strip()

def paraphrase_once(sentence, temp=0.75, style_hint=None):
    prompt = sentence if not style_hint else f"{sentence} (rewrite in {style_hint} style)"
    out = paraphraser(
        prompt,
        max_length=min(len(prompt.split()) + 20, 128),
        do_sample=True,
        temperature=temp,
        num_return_sequences=1,
        no_repeat_ngram_size=3
    )
    return clean_text_output(out[0]["generated_text"])

def multi_pass_rewrite(sentences, temp=0.7, style_hint=None, imperfection=0, passes=3):
    outputs = []
    for s in sentences:
        rewritten = s
        for _ in range(passes):
            rewritten = paraphrase_once(rewritten, temp=temp, style_hint=style_hint)
            rewritten = simplify_to_common(rewritten, extra_noise=True)

            # human-style noise: shuffle sentence breaks
            if random.random() < 0.25:
                rewritten = rewritten.replace(",", ".")  

        if imperfection > 0 and random.random() < imperfection / 8:
            rewritten = rewritten.capitalize()
        outputs.append(rewritten)
    return outputs

def split_sentences(text):
    return [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

def compute_ppl(text):
    enc = gpt2_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        loss = gpt2_model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item()) if loss.item() < 100 else float("inf")

def human_score_from_ppl(ppl):
    return int(max(0, min(100, 100 * (1 - (1 / (1 + math.exp(-((ppl - 15) / 20)))))))) if np.isfinite(ppl) else 0

def batch_similarity(original, variants):
    embeddings = embedder.encode([original] + variants, convert_to_numpy=True)
    orig_vec = embeddings[0]
    return [float(round(cosine_similarity([orig_vec], [embeddings[i+1]])[0][0] * 100, 2)) for i in range(len(variants))]

# -----------------------
# Style map
# -----------------------
style_map = {
    "Normal": (None, 0.7, 3),
    "Formal": ("formal", 0.55, 3),
    "Creative": ("creative", 0.9, 3),
    "Book Writer": ("descriptive and structured formal style suitable for writing books and modules", 0.65, 3)
}

# -----------------------
# UI
# -----------------------
st.title("🧩 AI Text Humanizer — Multi-Pass (Turnitin Mode)")
st.markdown(f"**Free tier usage:** {st.session_state.usage_count}/{FREE_TIER_LIMIT}")

with st.sidebar:
    st.header("Settings")
    style_choice = st.selectbox("Choose rewriting style:", list(style_map.keys()))
    imperfection = st.slider("Imperfection level", 0, 10, 6)
    passes = st.slider("Rewrite passes", 1, 4, 3)
    n_variants = st.slider("Variants", 1, 3, 1)
    input_text = st.text_area("Paste your text here:", height=300)

if st.button("🚀 Humanize Now"):
    if not input_text.strip():
        st.warning("Please paste some text first.")
    elif st.session_state.usage_count >= FREE_TIER_LIMIT:
        st.error("Free tier limit reached! Please upgrade.")
    else:
        st.session_state.usage_count += 1
        with st.spinner(f"Rewriting in {style_choice} style with {passes} passes..."):
            style_hint, temp, default_passes = style_map[style_choice]
            sentences = split_sentences(input_text)
            style_variants = []
            for v in range(n_variants):
                paraphrased_sents = multi_pass_rewrite(
                    sentences, temp=temp, style_hint=style_hint,
                    imperfection=imperfection, passes=passes
                )
                final_text = clean_text_output(" ".join(paraphrased_sents))
                style_variants.append(final_text)

            ppl_scores = [compute_ppl(txt) for txt in style_variants]
            human_scores = [human_score_from_ppl(p) for p in ppl_scores]
            sims = batch_similarity(input_text, style_variants)

            selected_var = st.selectbox(
                f"Select variant for {style_choice}:",
                [f"Variant {i+1}" for i in range(n_variants)],
                key=f"{style_choice}_select"
            )
            chosen_index = int(selected_var.split()[-1]) - 1

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.write(input_text)
            with col2:
                st.markdown(
                    f"**{selected_var}** — Score: {human_scores[chosen_index]}/100 "
                    f"| Similarity: {sims[chosen_index]}%"
                )
                st.write(style_variants[chosen_index])

            b = io.BytesIO(style_variants[chosen_index].encode("utf-8"))
            st.download_button(
                "Download", b,
                file_name=f"{style_choice}_{chosen_index+1}.txt",
                mime="text/plain"
            )
