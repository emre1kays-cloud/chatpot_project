import os
from langchain_community.vectorstores import Chroma  # Vektör veri deposu yönetimi için kullanılıyor
from langchain_community.embeddings import OpenAIEmbeddings  # OpenAI'nin gömleme (embedding) modeli
from pathlib import Path  # Dosya ve dizin işlemleri için kullanılıyor
from langchain.memory.buffer import ConversationBufferMemory  # Konuşma geçmişi yönetimi için bellek aracı
import json  # JSON formatında veri işlemek için kullanılıyor

# Kullanıcı hafızasının saklanacağı temel dizin
MEMORY_BASE_PATH = "./../user_memory"

# Temel dizinin mevcut olup olmadığını kontrol et, yoksa oluştur
Path(MEMORY_BASE_PATH).mkdir(parents=True, exist_ok=True)


def load_user_memory(user_id: str):
    """
    Kullanıcının hafızasını yükler veya yeni bir hafıza oluşturur.

    Args:
        user_id (str): Kullanıcıya özgü benzersiz kimlik numarası.

    Returns:
        ConversationBufferMemory: Kullanıcının konuşma geçmişini yöneten bellek nesnesi.
    """
    os.makedirs(MEMORY_BASE_PATH, exist_ok=True)  # Bellek dizini yoksa oluştur
    user_memory_path = os.path.join(MEMORY_BASE_PATH, f"{user_id}_memory_index")  # Kullanıcının hafızasının saklanacağı dosya yolu

    # Kullanıcının hafızası daha önce oluşturulmuş mu kontrol et
    if os.path.exists(user_memory_path):
        vector_store = Chroma(persist_directory=user_memory_path, embedding_function=OpenAIEmbeddings())  # Mevcut hafızayı yükle
    else:
        os.makedirs(user_memory_path, exist_ok=True)  # Dizin yoksa oluştur
        vector_store = Chroma(persist_directory=user_memory_path, embedding_function=OpenAIEmbeddings())  # Yeni hafıza oluştur

    stored_docs = vector_store.get()["documents"]  # Vektör deposundaki kayıtlı belgeleri al

    # Konuşma geçmişi yönetimi için bellek nesnesi oluştur
    memory = ConversationBufferMemory(vector_store=vector_store, memory_key="chat_history")

    # Hafızada kayıtlı belgeleri konuşma geçmişine ekle
    for i in range(0, len(stored_docs), 2):
        memory.save_context({"input": stored_docs[i]}, {"output": stored_docs[i+1]})

    return memory, vector_store  # Bellek nesnesini ve vektör deposunu döndür


def get_portfolio_from_memory(user_id: str) -> dict:
    """
    Kullanıcının portföy verilerini hafızadan alır.

    Args:
        user_id (str): Kullanıcıya özgü kimlik numarası.

    Returns:
        dict: Kullanıcının en güncel portföy verileri. Eğer veri bulunamazsa boş bir sözlük döner.
    """
    _, vector_store = load_user_memory(user_id)  # Kullanıcının hafızasını yükle

    # Vektör deposundan portföy verilerini arar
    results = vector_store.similarity_search(
        "PORTFOLIO_DATA_" + str(user_id),  
        k=1  # En güncel portföy verisini almak için en yüksek benzerlikteki 1 sonucu getir
    )

    # Portföy verisini kontrol et ve JSON formatına çevir
    for doc in results:
        if doc.page_content.startswith("PORTFOLIO_DATA_" + str(user_id)):
            try:
                json_str = doc.page_content.replace("PORTFOLIO_DATA_" + str(user_id), "").strip()  # Ön eki kaldır
                return json.loads(json_str)  # JSON formatına dönüştür ve döndür
            except json.JSONDecodeError:
                print("Error parsing portfolio data from vector store")  # JSON dönüştürme hatası durumunda mesaj yazdır

    return {}  # Eğer portföy verisi bulunamazsa boş sözlük döndür


def save_user_memory(user_id: str, input_text: str, output_text: str, portfolio_data: dict = None):
    """
    Kullanıcının konuşma geçmişini ve portföy verilerini hafızaya kaydeder.

    Args:
        user_id (str): Kullanıcıya özgü benzersiz kimlik numarası.
        input_text (str): Kullanıcının yazdığı giriş mesajı.
        output_text (str): Yapay zekanın ürettiği yanıt.
        portfolio_data (dict, optional): Eğer mevcutsa, kullanıcının portföy verileri.
    """
    user_memory, vector_store = load_user_memory(user_id)  # Kullanıcı hafızasını yükle

    # Kullanıcının girdisini ve yapay zekanın yanıtını hafızaya ekle
    user_memory.save_context({"input": input_text}, {"output": output_text})

    # Konuşma geçmişini vektör deposuna ekle
    vector_store.add_texts([input_text, output_text])

    # Eğer portföy verisi mevcutsa, onu özel bir anahtar ile sakla
    if portfolio_data:
        portfolio_marker = "PORTFOLIO_DATA"  # Portföy verileri için özel bir anahtar
        portfolio_text = f"{portfolio_marker}_{user_id} {json.dumps(portfolio_data)}"  # JSON formatında sakla
        vector_store.add_texts([f"{user_id}_JSON_PORTFOLIO", portfolio_text])  # Vektör deposuna ekle

        # Konuşma belleğine de portföy verisini ekleyerek, bağlamın korunmasını sağla
        user_memory.save_context(
            {"input": "System: Storing portfolio data"},
            {"output": portfolio_text}
        )

    # Verileri kalıcı olarak sakla
    vector_store.persist()
    print(user_memory.load_memory_variables({}))  # Bellekte saklanan verileri yazdır
