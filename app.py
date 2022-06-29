#!C:\Anaconda2020\Anaconda3\python.exe
# import sys
import os
# import io
# import gc
import warnings
import pickle
# from IPython.display import display

from caption_utils import load_models, image_load, image_process, get_captions#, image_show

import streamlit as st

### PATHS
UTILITIES_PATH = './utilities/'
DATA_PATH = './data/'

### PATHS TO MODELS
CAPTION_MODEL_PATH = UTILITIES_PATH + 'caption_net.pth'
# INCEPTION_PATH = UTILITIES_PATH + 'bh_inception.pth'

TMP_DIR = UTILITIES_PATH + '/tmp/'
TMP_IMG_PATH = TMP_DIR + 'img.jpg'

new_image = False


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def instantiate():
    with open(UTILITIES_PATH + 'word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    with open(UTILITIES_PATH + 'idx2word.pkl', 'rb') as f:
        idx2word = pickle.load(f)
    with open(UTILITIES_PATH + 'spec_tokens.pkl', 'rb') as f:
        spec_tokens = pickle.load(f)

    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN = (spec_tokens[x] for x in
                                                  ['BOS_token', 'EOS_token', 'PAD_token', 'UNK_token'])
    BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX = (word2idx[spec_tokens[x]] for x in
                                          ['BOS_token', 'EOS_token', 'PAD_token', 'UNK_token'])
    VOCAB_SIZE = len(word2idx)

    message_slot = st.empty()
    with warnings.catch_warnings():
        message_slot.info('loading models, this might take some time..')
        warnings.simplefilter('ignore', FutureWarning)
        inception, caption_model, device = load_models(CAPTION_MODEL_PATH, VOCAB_SIZE, PAD_IDX)
        message_slot.empty()

    EXCLUDE_FROM_PREDICTION = [BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX]

    return inception, caption_model, idx2word, BOS_IDX, EOS_IDX, EXCLUDE_FROM_PREDICTION


@st.cache(suppress_st_warning=True, ttl=3600, max_entries=1, show_spinner=False)
def load_image(img):
    global new_image
    new_image = True

    if isinstance(img, str):
        image = image_load(img)
    else:
        img_bytes = img.read()
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        with open(TMP_IMG_PATH, 'wb') as f:
            f.write(img_bytes)
        # image = image_load(io.BytesIO(img_bytes))
        image = image_load(TMP_IMG_PATH)
    return image


def main():
    st.set_page_config(page_title='Small, but awesome image captioning tool demo',
                       page_icon=None, layout='wide', initial_sidebar_state='expanded')
    st.title('Image Captioning Tool Demo\nSmall, but awesome')

    spinner_slot = st.empty()
    left, right = st.beta_columns((1, 1))
    image_slot = left
    caption_slot = right

    inception, caption_model, idx2word, BOS_IDX, EOS_IDX, EXCLUDE_FROM_PREDICTION = instantiate()

    is_loaded = False
    img_loaded = st.file_uploader(label='Upload your image in .jpg format', type=['jpg', 'jpeg'])
    load_status_slot = st.empty()

    if img_loaded:
        img = load_image(img_loaded)
        is_loaded = True
        img, img_for_net, img_size_init = image_process(img)
        image_slot.image(img, use_column_width=False, width=600)
        if new_image:
            load_status_slot.success('Image loaded!')

    # MAX_CAPTION_LEN = 10 
    MAX_CAPTION_LEN = st.sidebar.number_input("Select maximal caption length:",
                                              min_value=1, max_value=20,
                                              value=8, step=1)
    # SAMPLING = True
    SAMPLING = st.sidebar.radio("Should sampling be done?", ('Sampling', 'Use Best Option'))
    SAMPLE = SAMPLING == 'Sampling'

    n_captions_slot = st.sidebar.empty()
    slider_slot = st.sidebar.empty()

    if SAMPLE:
        N_CAPTIONS = n_captions_slot.number_input("Select number of captions generated:",
                                                  min_value=1, max_value=15, value=5, step=1)
        TEMPERATURE = slider_slot.slider("Set temperature for sampling: ",
                                         min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    else:
        N_CAPTIONS = 1
        TEMPERATURE = 1
        slider_slot.markdown('No sampling would be made while generating the one and only caption variant')

    button_slot = st.sidebar.empty()
    warning_slot = st.sidebar.empty()

    if button_slot.button('Generate Captions!'):
        load_status_slot.empty()
        if is_loaded:
            # with spinner_slot.spinner('Generating...'):
            spinner_slot.info('Generating...')
            generated_captions = get_captions(img_for_net, inception, caption_model, idx2word,
                                              EXCLUDE_FROM_PREDICTION, (BOS_IDX,), EOS_IDX,
                                              N_CAPTIONS, TEMPERATURE, SAMPLE, MAX_CAPTION_LEN)
            spinner_slot.empty()
            caption_slot.header('What are we seeing there..')
            caption_slot.markdown('  \n'.join(generated_captions))
        else:
            warning_slot.warning('Please, upload your image first')

    st.sidebar.markdown("""\
    <span style="color:black;font-size:8"><p>made by\
    <a style="color:mediumorchid" href="https://data.mail.ru/profile/a.nalitkin/">aleksandr</a>
    &
    <a style="color:crimson" href="https://data.mail.ru/profile/m.korotkov/">michael</a>
    </p></span>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
