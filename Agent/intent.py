import os  # İşletim sistemi ile ilgili işlemleri yapmak için kullanılır.
from langchain_community.vectorstores import Chroma  # Chroma vektör veritabanını kullanmak için.
from langchain_community.embeddings import OpenAIEmbeddings  # Metinleri vektörlere dönüştürmek için OpenAI gömlemeleri.
from pathlib import Path  # Dosya ve dizin yollarını yönetmek için kullanılır.
from langchain.memory.buffer import ConversationBufferMemory  # Konuşma geçmişini saklayan bellek yönetimi sınıfı.
import json  # JSON formatında veri işlemek için.

# Kullanıcı bellek verilerinin saklanacağı dizin yolu.
MEMORY_BASE_PATH = "./../user_memory"

# Belirtilen dizinin mevcut olup olmadığını kontrol eder, yoksa oluşturur.
Path(MEMORY_BASE_PATH).mkdir(parents=True, exist_ok=True)

def load_user_memory(user_id: str):
    """
    Belirtilen kullanıcı kimliği için Chroma vektör deposunu yükler veya oluşturur.
    
    Args:
        user_id (str): Kullanıcının benzersiz kimliği.

    Returns:
        ConversationBufferMemory: Kullanıcının konuşma geçmişini yöneten bellek nesnesi.
        Chroma: Kullanıcıya ait vektör tabanlı bellek deposu.
    """
    
    # Kullanıcının bellek dosyalarının saklanacağı dizin yolu.
    user_memory_path = os.path.join(MEMORY_BASE_PATH, f"{user_id}_memory_index")

    # Chroma vektör deposunu yükle veya oluştur.
    if os.path.exists(user_memory_path):
        vector_store = Chroma(persist_directory=user_memory_path, embedding_function=OpenAIEmbeddings())
    else:
        os.makedirs(user_memory_path, exist_ok=True)  # Dizin yoksa oluştur.
        vector_store = Chroma(persist_directory=user_memory_path, embedding_function=OpenAIEmbeddings())

    # Bellekte saklanan tüm dökümanları al.
    stored_docs = vector_store.get()["documents"]

    # Kullanıcı için konuşma belleği nesnesini oluştur.
    memory = ConversationBufferMemory(vector_store=vector_store, memory_key="chat_history")

    # Saklanan dökümanları çiftler halinde belleğe ekle (girdi-çıktı eşleşmeleri).
    for i in range(0, len(stored_docs), 2):
        memory.save_context({"input": stored_docs[i]}, {"output": stored_docs[i+1]})

    return memory, vector_store  # Kullanıcıya ait bellek ve vektör deposunu döndür.


def get_portfolio_from_memory(user_id: str) -> dict:
    """
    Kullanıcının en son kaydedilen portföy verilerini vektör deposundan getirir.

    Args:
        user_id (str): Kullanıcının benzersiz kimliği.

    Returns:
        dict: Kullanıcının en güncel portföy verileri. Eğer bulunamazsa boş bir sözlük döner.
    """

    _, vector_store = load_user_memory(user_id)  # Kullanıcının vektör deposunu yükle.

    # Kullanıcının portföy verilerini en son kaydedilen versiyonuna göre ara.
    results = vector_store.similarity_search(
        "PORTFOLIO_DATA_" + str(user_id),  # Kullanıcıya özel bir anahtar kelime ile arama yap.
        k=1  # En güncel (en alakalı) sonucu almak için 1 adet veri getir.
    )

    # Sonuçlar içinde portföy verisini kontrol et.
    for doc in results:
        if doc.page_content.startswith("PORTFOLIO_DATA_" + str(user_id)):
            try:
                # JSON formatındaki veriyi temizleyip dönüştür.
                json_str = doc.page_content.replace("PORTFOLIO_DATA_" + str(user_id), "").strip()
                return json.loads(json_str)  # JSON stringini sözlük formatına çevir ve döndür.
            except json.JSONDecodeError:
                print("Vektör deposundan alınan portföy verisi çözümlenemedi.")

    return {}  # Portföy verisi bulunamazsa boş bir sözlük döndür.


def save_user_memory(user_id: str, input_text: str, output_text: str, portfolio_data: dict = None):
    """
    Kullanıcının konuşma geçmişini ve isteğe bağlı olarak portföy verilerini kaydeder.

    Args:
        user_id (str): Kullanıcının benzersiz kimliği.
        input_text (str): Kullanıcının girdiği mesaj.
        output_text (str): Sistem tarafından oluşturulan yanıt.
        portfolio_data (dict, optional): Kaydedilecek portföy verileri (varsa).
    """

    user_memory, vector_store = load_user_memory(user_id)  # Kullanıcının belleğini ve vektör deposunu yükle.

    # Kullanıcının girişini ve sistemin yanıtını bellek tamponuna kaydet.
    user_memory.save_context({"input": input_text}, {"output": output_text})

    # Konuşma geçmişini vektör deposuna ekle.
    vector_store.add_texts([input_text, output_text])

    # Eğer portföy verisi varsa, bunu vektör deposuna özel bir etiketle kaydet.
    if portfolio_data:
        portfolio_marker = "PORTFOLIO_DATA"
        portfolio_text = f"{portfolio_marker}_{user_id} {json.dumps(portfolio_data)}"

        # Portföy verisini vektör deposuna ekle.
        vector_store.add_texts([f"{user_id}_JSON_PORTFOLIO", portfolio_text])

        # Konuşma geçmişine de ekleyerek, bağlam içinde kullanılmasını sağla.
        user_memory.save_context(
            {"input": "System: Portföy verisi saklanıyor."},
            {"output": portfolio_text}
        )

    # Değişiklikleri vektör deposuna kalıcı olarak kaydet.
    vector_store.persist()

    # Güncellenmiş bellek değişkenlerini ekrana yazdır (debug amaçlı).
    print(user_memory.load_memory_variables({}))
