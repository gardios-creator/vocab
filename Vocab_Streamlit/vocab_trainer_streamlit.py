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

@st.cache_data
def load_data():
    """Load and sanitize data from Dropbox."""
    try:
        _, res = dbx.files_download(FILE_PATH)
        with io.BytesIO(res.content) as f:
            df = pd.read_excel(f, engine='openpyxl')
    except:
        df = pd.DataFrame(columns=['Language', 'Original', 'Translation', 'Hint', 'Category', 'Attempts', LAST_ASKED_COL, STAGE_COL])
    
    df = df.fillna('')
    
    if LAST_ASKED_COL not in df.columns:
        df[LAST_ASKED_COL] = ''
    if STAGE_COL not in df.columns:
        df[STAGE_COL] = 0
    
    df[LAST_ASKED_COL] = pd.to_datetime(df[LAST_ASKED_COL], errors='coerce').dt.date
    df[STAGE_COL] = pd.to_numeric(df[STAGE_COL], errors='coerce').fillna(0).astype(int)
    
    return df

def save_data(df):
    """Save current state to Dropbox."""
    df_to_save = df[df['Category'] != 'delete'].sort_values(by=SORTING_COLS).copy()
    for col in ['Original', 'Translation', 'Hint']:
        df_to_save[col] = df_to_save[col].astype(str).str.strip()
    
    df_to_save[LAST_ASKED_COL] = df_to_save[LAST_ASKED_COL].astype(str)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_to_save.to_excel(writer, index=False)
    
    dbx.files_upload(output.getvalue(), FILE_PATH, mode=dropbox.files.WriteMode.overwrite)

def increment_progress(idx):
    """Update attempt counters and spaced-repetition stages."""
    if st.session_state.stage_updated:
        return
    
    df = st.session_state.df
    current_cat = df.loc[idx, 'Category']
    current_stage = df.loc[idx, STAGE_COL]
    
    df.at[idx, 'Attempts'] = int(df.loc[idx, 'Attempts'] or 0) + 1
    df.at[idx, LAST_ASKED_COL] = date.today()
    
    if current_cat in ACQUIRED_CATEGORIES:
        new_stage = 0
    else:
        new_stage = (current_stage + 1) % N_STAGES
    
    df.at[idx, STAGE_COL] = new_stage
    st.session_state.stage_updated = True

def get_random_instance():
    """Sample next word based on selection criteria."""
    st.session_state.sampled_index = None
    st.session_state.stage_updated = False
    st.session_state.show_original = False
    st.session_state.show_translation = False
    st.session_state.display_hint = False
    st.session_state.show_update = False
    
    df = st.session_state.df
    filter_ = (df['Language'] == st.session_state.lang_choice)
    if st.session_state.cat_choice != "all":
        filter_ = filter_ & (df['Category'] == st.session_state.cat_choice)
    
    df_cond = df[filter_].copy()
    if df_cond.empty:
        return

    if st.session_state.mode_choice == "Didactical":
        today = date.today()
        df_cond['Age'] = (today - df_cond[LAST_ASKED_COL]).apply(lambda x: x.days if pd.notna(x) else 365).clip(lower=0)
        
        review_mask = pd.Series(False, index=df_cond.index)
        for i_st in range(N_STAGES):
            stage_number = i_st + 1
            required_age = DAYS_STAGES[i_st]
            current_stage_mask = (df_cond[STAGE_COL] == stage_number) & (df_cond['Age'] > required_age)
            review_mask = review_mask | current_stage_mask
        
        df_review = df_cond[review_mask]
        df_new = df_cond[df_cond[STAGE_COL] == 0]
        
        if not df_review.empty or not df_new.empty:
            p_review = REVIEW_STAGE_WEIGHT if not df_review.empty else 0
            p_new = NEW_STAGE_WEIGHT if not df_new.empty else 0
            chosen_pool = np.random.choice(['rev', 'new'], p=[p_review/(p_review+p_new), p_new/(p_review+p_new)])
            final_pool = df_review if chosen_pool == 'rev' else df_new
        else:
            final_pool = df_cond
        
        weights = final_pool['Age'].clip(lower=1)
        st.session_state.sampled_index = np.random.choice(final_pool.index, p=weights/weights.sum())
    else:
        st.session_state.sampled_index = np.random.choice(df_cond.index)
    
    idx = st.session_state.sampled_index
    if df.loc[idx, 'Category'] in PASSIVE_CATEGORIES:
        st.session_state.show_original = True
    else:
        st.session_state.show_translation = True
    
    increment_progress(idx)
    st.session_state.counter_tested += 1

# --- UI Setup ---
st.set_page_config(page_title="Vocab", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'sampled_index' not in st.session_state:
    st.session_state.sampled_index = None
if 'counter_tested' not in st.session_state:
    st.session_state.counter_tested = 0

for flag in ['stage_updated','show_original','show_translation','show_update','display_hint']:
    if flag not in st.session_state: st.session_state[flag] = False

# --- Sidebar ---
st.session_state.lang_choice = st.sidebar.selectbox("Language", sorted(st.session_state.df['Language'].unique().tolist()))
st.session_state.cat_choice = st.sidebar.selectbox("Category", ["all"] + sorted(st.session_state.df['Category'].unique().tolist()))
st.session_state.mode_choice = st.sidebar.selectbox("Mode", ["Didactical", "Uniform"])

if st.sidebar.button("💾 Save Database", use_container_width=True):
    save_data(st.session_state.df)
    st.sidebar.success("Database Saved!")

# --- Main Display ---
st.button("🎲 NEXT WORD", use_container_width=True, on_click=get_random_instance)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("👁️ Orig", use_container_width=True): st.session_state.show_original = True
with col2:
    if st.button("👁️ Trans", use_container_width=True): st.session_state.show_translation = True
with col3:
    if st.button("💡 Hint", use_container_width=True): st.session_state.display_hint = True
with col4:
    if st.button("✏️ Edit", use_container_width=True): st.session_state.show_update = not st.session_state.show_update

if st.session_state.sampled_index is not None:
    idx = st.session_state.sampled_index
    row = st.session_state.df.loc[idx]
    
    display_style = 'font-size:24px; white-space:pre-wrap; font-weight:bold; margin-top:15px;'
    
    if st.session_state.show_original:
        st.markdown(f'<p style="{display_style} color:#3366ff;">{row["Original"]}</p>', unsafe_allow_html=True)
    
    if st.session_state.show_translation:
        st.markdown(f'<p style="{display_style} color:#2eb82e;">{row["Translation"]}</p>', unsafe_allow_html=True)
        
    if st.session_state.display_hint:
        hint_content = str(row['Hint']).strip()
        if hint_content:
            st.info(hint_content)

    if st.session_state.show_update:
        with st.form("edit_entry"):
            u_orig = st.text_area("Original", value=row['Original'])
            u_trans = st.text_area("Translation", value=row['Translation'])
            u_hint = st.text_input("Hint", value=row['Hint'])
            
            all_cats = sorted(st.session_state.df['Category'].unique().tolist())
            u_cat = st.selectbox("Category", all_cats, index=all_cats.index(row['Category']) if row['Category'] in all_cats else 0)
            u_stage = st.number_input("Stage", 0, N_STAGES-1, value=int(row[STAGE_COL]))
            
            if st.form_submit_button("Update"):
                st.session_state.df.at[idx, 'Original'] = u_orig
                st.session_state.df.at[idx, 'Translation'] = u_trans
                st.session_state.df.at[idx, 'Hint'] = u_hint
                st.session_state.df.at[idx, 'Category'] = u_cat
                if u_cat in ACQUIRED_CATEGORIES:
                    st.session_state.df.at[idx, STAGE_COL] = 0
                else:
                    st.session_state.df.at[idx, STAGE_COL] = u_stage
                st.success("Entry updated!")


    st.caption(f"Cat: {row['Category']} | Stage: {row[STAGE_COL]} | Attempts: {row['Attempts']} | Session total: {st.session_state.counter_tested}")
else:
    st.write("Click 'NEXT WORD' to begin.")
