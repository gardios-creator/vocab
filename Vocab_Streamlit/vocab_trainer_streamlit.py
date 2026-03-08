import streamlit as st
import pandas as pd
import dropbox
import io
import numpy as np
from datetime import date, timedelta

# --- Configuration ---
APP_KEY = st.secrets['APP_KEY']
APP_SECRET = st.secrets['APP_SECRET']
REFRESH_TOKEN = st.secrets['REFRESH_TOKEN']
FILE_PATH = "/VocTrainer/Vocab_DB.xlsx"



# --- Constants ---
LAST_ASKED_COL = 'Last_Asked'
STAGE_COL = 'Stage'
DAYS_STAGES = [0, 7, 15, 30, 60, 120, 360]
N_STAGES = len(DAYS_STAGES)
ACQUIRED_CATEGORIES = {'02 - acquired', '06 - grammar acquired', '08 - gender acquired', '10 - math acquired'}
PASSIVE_CATEGORIES = {'00 - unknown', '15 - collocations'}
REVIEW_STAGE_WEIGHT = 6
NEW_STAGE_WEIGHT = 4
SORTING_COLS = ['Language', 'Category', 'Attempts']

dbx = dropbox.Dropbox(
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth2_refresh_token=REFRESH_TOKEN
)

# --- Data Loading and Saving ---
@st.cache_data
def load_data():
    """Load data from Dropbox."""
    try:
        _, res = dbx.files_download(FILE_PATH)
        with io.BytesIO(res.content) as f:
            df = pd.read_excel(f, engine='openpyxl')
    except:
        df = pd.DataFrame(columns=['Language', 'Original', 'Translation', 'Hint', 'Category', 'Attempts', LAST_ASKED_COL, STAGE_COL])
    
    # Initialize columns
    if LAST_ASKED_COL not in df.columns:
        df[LAST_ASKED_COL] = ''
    if STAGE_COL not in df.columns:
        df[STAGE_COL] = 0
    
    # Process dates and stages
    df[LAST_ASKED_COL] = pd.to_datetime(df[LAST_ASKED_COL], errors='coerce').dt.date
    one_year_ago = date.today() - timedelta(days=365)
    missing_mask = df[LAST_ASKED_COL].isna()
    if missing_mask.any():
        df.loc[missing_mask, LAST_ASKED_COL] = one_year_ago
    
    df[STAGE_COL] = pd.to_numeric(df[STAGE_COL], errors='coerce').fillna(0).astype(int)
    
    return df

def save_data(df):
    """Save data to Dropbox."""
    df_to_save = df[df['Category'] != 'delete'].sort_values(by=SORTING_COLS).copy()
    for col in ['Original', 'Translation', 'Hint']:
        df_to_save[col] = df_to_save[col].str.strip()
    df_to_save[LAST_ASKED_COL] = df_to_save[LAST_ASKED_COL].astype(str)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_save.to_excel(writer, index=False)
    
    dbx.files_upload(output.getvalue(), FILE_PATH, mode=dropbox.files.WriteMode.overwrite)

# --- Page Config ---
st.set_page_config(page_title="Vocabulary Trainer", layout="wide")
st.title("📚 Vocabulary Trainer")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'sampled_index' not in st.session_state:
    st.session_state.sampled_index = None
# flags for what to display
for flag in ['stage_updated','show_original','show_translation','show_update','display_hint']:
    if flag not in st.session_state:
        st.session_state[flag] = False
if 'counter_tested' not in st.session_state:
    st.session_state.counter_tested = 0

df = st.session_state.df

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
language = st.sidebar.selectbox("Language", ["Select Language"] + sorted(df['Language'].unique().tolist()), key="lang")
category = st.sidebar.selectbox("Category", ["all"] + sorted(df['Category'].unique().tolist()), key="cat")
temperature = st.sidebar.selectbox("Sampling Mode", ["Didactical", "Uniform", "Only unseen", "Only seen"], key="temp")

# --- Helper Functions ---
def get_random_instance():
    """Sample a random vocabulary entry."""
    if language == "Select Language":
        st.error("Please select a language!")
        return
    
    st.session_state.sampled_index = None
    # reset display toggles
    st.session_state.stage_updated = False
    st.session_state.show_original = False
    st.session_state.show_translation = False
    st.session_state.display_hint = False
    st.session_state.show_update = False
    
    filter_ = (df['Language'] == language)
    if category != "all":
        filter_ = filter_ & (df['Category'] == category)
    
    df_cond = df[filter_].copy()
    
    if df_cond.empty:
        st.error(f"No entries available for {language} in category {category}.")
        return
    
    # Ensure dates are date objects
    df_cond[LAST_ASKED_COL] = pd.to_datetime(df_cond[LAST_ASKED_COL], errors='coerce').dt.date
    
    # Didactical sampling
    if temperature == "Didactical":
        today = date.today()
        df_cond['Age'] = (today - df_cond[LAST_ASKED_COL]).apply(lambda x: x.days if pd.notna(x) else 0).clip(lower=0)
        
        # Review pool
        review_mask = pd.Series(False, index=df_cond.index)
        for i_st in range(N_STAGES):
            stage_number = i_st + 1
            required_age = DAYS_STAGES[i_st]
            current_stage_mask = (df_cond[STAGE_COL] == stage_number) & (df_cond['Age'] > required_age)
            review_mask = review_mask | current_stage_mask
        
        df_review = df_cond[review_mask].copy()
        df0 = df_cond[df_cond[STAGE_COL] == 0].copy()
        
        has_review = not df_review.empty
        has_new = not df0.empty
        
        if has_review or has_new:
            pool_probs = {}
            if has_review:
                pool_probs['review'] = REVIEW_STAGE_WEIGHT
            if has_new:
                pool_probs['new'] = NEW_STAGE_WEIGHT
            
            total = sum(pool_probs.values())
            pool_probs = {k: v / total for k, v in pool_probs.items()}
            
            chosen_pool = np.random.choice(list(pool_probs.keys()), p=list(pool_probs.values()))
            final_pool = df_review if chosen_pool == 'review' else df0
        else:
            final_pool = df_cond
        
        # Age weighting
        final_pool['Effective_Age'] = final_pool['Age'].clip(lower=1)
        total_age_weight = final_pool['Effective_Age'].sum()
        
        if total_age_weight > 0:
            probs = final_pool['Effective_Age'] / total_age_weight
        else:
            probs = np.repeat(1.0, len(final_pool)) / len(final_pool)
        
        probs = probs.reindex(final_pool.index, fill_value=0)
        st.session_state.sampled_index = np.random.choice(final_pool.index, p=probs)
    
    # Other sampling modes
    else:
        max_attempts = df_cond['Attempts'].max() if not df_cond.empty else 0
        
        if temperature == "Only seen" and max_attempts > 0:
            probs = df_cond['Attempts'] / df_cond['Attempts'].sum()
        elif temperature == "Only unseen":
            df_unseen = df_cond[df_cond['Attempts'] == 0]
            if not df_unseen.empty:
                probs = df_cond.index.map(lambda i: 1 / len(df_unseen) if i in df_unseen.index else 0)
                probs = probs[probs > 0] / probs[probs > 0].sum()
                df_cond = df_cond[probs > 0]
            else:
                probs = np.repeat(1.0, len(df_cond)) / len(df_cond)
        else:  # Uniform
            probs = np.repeat(1.0, len(df_cond)) / len(df_cond)
        
        probs = probs.values if hasattr(probs, 'values') else probs
        st.session_state.sampled_index = np.random.choice(df_cond.index, p=probs)
    
    # set default visible side according to category
    if st.session_state.sampled_index is not None:
        cat = df.loc[st.session_state.sampled_index, 'Category']
        if cat in PASSIVE_CATEGORIES:
            st.session_state.show_original = True
        else:
            st.session_state.show_translation = True
        # count as attempt when word automatically shown
        increment_attempts_and_stage(st.session_state.sampled_index)
    st.session_state.counter_tested += 1

def increment_attempts_and_stage(idx):
    """Update attempts, date, and stage for current word."""
    if st.session_state.stage_updated:
        return
    
    current_category = df.loc[idx, 'Category']
    current_stage = df.loc[idx, STAGE_COL]
    
    df.loc[idx, 'Attempts'] = df.loc[idx, 'Attempts'] + 1
    df.loc[idx, LAST_ASKED_COL] = date.today()
    
    if current_category in ACQUIRED_CATEGORIES:
        new_stage = 0
    else:
        new_stage = (current_stage + 1) % N_STAGES
    
    df.loc[idx, STAGE_COL] = new_stage
    st.session_state.stage_updated = True

# --- Main UI ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🎲 Next Word", use_container_width=True):
        get_random_instance()

available_count = len(df[(df['Language'] == language) & ((category == "all") | (df['Category'] == category))])
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Available Entries", available_count)
with col2:
    st.metric("Exercises Done", st.session_state.counter_tested)
with col3:
    if st.session_state.sampled_index is not None:
        current_stage = df.loc[st.session_state.sampled_index, STAGE_COL]
        st.metric("Current Stage", current_stage)

# --- Display Current Word ---
if st.session_state.sampled_index is not None:
    idx = st.session_state.sampled_index
    category_sampled = df.loc[idx, 'Category']
    
    st.divider()
    st.subheader("📖 Current Word")
    st.markdown(f"**Category:** {category_sampled}")
    
    # display buttons for revealing content
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Show Original", use_container_width=True, key="show_orig"):
            increment_attempts_and_stage(idx)
            st.session_state.show_original = True
        if st.button("💡 Show Translation", use_container_width=True, key="show_trans"):
            increment_attempts_and_stage(idx)
            st.session_state.show_translation = True
        if st.button("� Show Hint", use_container_width=True, key="show_hint"):
            increment_attempts_and_stage(idx)
            st.session_state.display_hint = True
    with col2:
        if st.button("✏️ Update Entry", use_container_width=True, key="update"):
            st.session_state.show_update = True
    
    # actual text display
    if st.session_state.show_original:
        st.info(f"**Original:** {df.loc[idx, 'Original']}")
    if st.session_state.show_translation:
        st.success(f"**Translation:** {df.loc[idx, 'Translation']}")
    if st.session_state.display_hint:
        hint_text = df.loc[idx, 'Hint']
        if pd.notna(hint_text) and str(hint_text).strip():
            st.info(f"**Hint:** {hint_text}")
    
    # update form remains unchanged
    if st.session_state.show_update:
        with st.form("update_form", border=False):
            new_orig = st.text_area("Original:", value=df.loc[idx, 'Original'], key="edit_orig")
            new_trans = st.text_area("Translation:", value=df.loc[idx, 'Translation'], key="edit_trans")
            new_hint = st.text_input("Hint:", value=df.loc[idx, 'Hint'], key="edit_hint")
            new_cat = st.selectbox("Category:", sorted(df['Category'].unique()), index=list(df['Category'].unique()).index(df.loc[idx, 'Category']), key="edit_cat")
            new_stage = st.slider("Stage:", 0, N_STAGES - 1, df.loc[idx, STAGE_COL], key="edit_stage")
            
            if st.form_submit_button("Save Changes"):
                df.loc[idx, 'Original'] = new_orig.strip()
                df.loc[idx, 'Translation'] = new_trans.strip()
                df.loc[idx, 'Hint'] = new_hint.strip()
                df.loc[idx, 'Category'] = new_cat
                df.loc[idx, STAGE_COL] = new_stage
                st.success("Entry updated!")
                st.session_state.show_update = False


# --- Save Database ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save to Dropbox", use_container_width=True, key="save"):
        save_data(df)
        st.success("Database saved to Dropbox!")
