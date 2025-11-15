from langchain_openai import ChatOpenAI

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import streamlit as st
from streamlit_chat import message

# loading the OpenAI api key from .env (OPENAI_API_KEY="sk-********")
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title='Eczane SUT Kontrol',
    page_icon='🤖'
)
st.subheader('Eczane AI SGK Ödeme Kontrol Robotu🤖')

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

# creating the messages (chat history) in the Streamlit session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# creating the sidebar
with st.sidebar:
    st.title('⚙️ Ayarlar')

    # streamlit text input widget for the system message (role)
    system_message = st.text_area(label='System role', height=80)

    # streamlit password input widget for the password
    password = st.text_input(label='Şifre', type='password')

    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(content=system_message)
            )

    # Reçete Sahibi Bilgileri
    st.divider()
    st.subheader('👤 Reçete Sahibi Bilgileri')

    patient_name = st.text_input(label='Hasta Adı Soyadı')
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input(label='Yaş', min_value=0, max_value=150, value=0)
    with col2:
        patient_gender = st.selectbox(label='Cinsiyet', options=['Erkek', 'Kadın', 'Diğer'])

    # Geçmiş Raporlar
    st.divider()
    st.subheader('📋 Geçmiş Raporlar')

    if 'past_reports' not in st.session_state:
        st.session_state.past_reports = []

    # Rapor ekleme alanı
    new_report = st.text_area(label='Yeni Rapor Ekle', height=80, placeholder='Rapor yazısını buraya yapıştırınız...')
    if st.button('Rapor Ekle', key='add_report'):
        if new_report:
            st.session_state.past_reports.append(new_report)
            st.success('Rapor eklendi!')

    # PDF yükleme alanı
    uploaded_pdfs = st.file_uploader(label='Raporları PDF Olarak Toplu Yükle', type=['pdf'], accept_multiple_files=True)
    if uploaded_pdfs:
        for pdf_file in uploaded_pdfs:
            st.info(f'📄 {pdf_file.name} yüklendi')

    # Geçmiş raporları listele
    if st.session_state.past_reports:
        st.write('**Kaydedilen Raporlar:**')
        for idx, report in enumerate(st.session_state.past_reports, 1):
            with st.expander(f'Rapor {idx}'):
                st.write(report)
                if st.button(f'Sil', key=f'delete_report_{idx}'):
                    st.session_state.past_reports.pop(idx - 1)
                    st.rerun()

    # Geçmiş İlaçlar
    st.divider()
    st.subheader('💊 Geçmiş İlaçlar')

    if 'past_medications' not in st.session_state:
        st.session_state.past_medications = []

    col1, col2 = st.columns([3, 1])
    with col1:
        medication_name = st.text_input(label='İlaç Adı')
    with col2:
        medication_date = st.date_input(label='Tarih')

    if st.button('İlaç Ekle', key='add_medication'):
        if medication_name:
            st.session_state.past_medications.append({
                'name': medication_name,
                'date': medication_date.strftime('%d.%m.%Y')
            })
            st.success('İlaç eklendi!')

    # Geçmiş ilaçları listele
    if st.session_state.past_medications:
        st.write('**İlaç Geçmişi:**')
        for idx, med in enumerate(st.session_state.past_medications, 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{idx}. {med['name']}")
            with col2:
                st.caption(med['date'])
            with col3:
                if st.button(f'❌', key=f'delete_med_{idx}'):
                    st.session_state.past_medications.pop(idx - 1)
                    st.rerun()

# displaying the messages (chat history)
for i, msg in enumerate(st.session_state.messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=f'{i} + 🤓')  # user's question
    else:
        message(msg.content, is_user=False, key=f'{i} +  🤖')  # ChatGPT response

# Reçete Verileri Giriş Bölümü
st.divider()
st.subheader('📝 Reçete Verileri')

col1, col2 = st.columns(2)

with col1:
    st.markdown('**Rapor Yazısı**')
    report_text = st.text_area(
        label='Reçetedeki rapor yazısını giriniz',
        height=120,
        placeholder='Rapor yazısını buraya yapıştırınız...',
        key='prescription_report'
    )

with col2:
    st.markdown('**İlaç Kodları**')
    drug_codes = st.text_area(
        label='İlaç kodlarını giriniz',
        height=120,
        placeholder='İlaç kodlarını satır satır giriniz (her kod ayrı satırda)',
        key='drug_codes'
    )

# adding a default SystemMessage if the user didn't entered one
if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))

# User input at the bottom
st.divider()
col1, col2 = st.columns([5, 1])

with col1:
    user_prompt = st.text_input(label='Mesaj gönder', placeholder='Buraya yazınız...')

with col2:
    send_button = st.button('➤ Gönder', use_container_width=True)

# if the user entered a question or clicked the send button
if user_prompt or send_button:
    if user_prompt:  # Check if there's actually text to send
        # Reçete verilerini mesaja ekle
        full_message = user_prompt

        if report_text or drug_codes:
            full_message += "\n\n---\n**Reçete Bilgileri:**\n"
            if report_text:
                full_message += f"**Rapor:** {report_text}\n"
            if drug_codes:
                full_message += f"**İlaç Kodları:** {drug_codes}"

        st.session_state.messages.append(
            HumanMessage(content=full_message)
        )

        with st.spinner('Yanıt hazırlanıyor ...'):
            # creating the ChatGPT response
            response = chat(st.session_state.messages)

        # adding the response's content to the session state
        st.session_state.messages.append(AIMessage(content=response.content))

        # Rerun the app to display the new messages
        st.rerun()

# run the app: streamlit run ./project_streamlit_custom_chatgpt.py