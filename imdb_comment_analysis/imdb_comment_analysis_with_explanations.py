from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb


"""
en çok kullanılan 10k kelimeyi dikkate alacağız çünkü modele her kelimeyi anlamlı anlamsız 
eklersek model karmaşıklığı artar ve overfitting e sebep olur en çok anlamlı 10k kelimeyi ekledik
"""
max_features=10000

"""
Hazır veri setlerinde dataset eklendiğinde direk (x_train,y_train),(x_test,y_test)
şeklinde return eder kendini o yüzden değişken tanımlamalarını bu şekilde yapıyoruz
"""
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
#num_words yani kendinden farklı kaç adet kelime entegre edeceğimizi de ekledik
#bu özellik çıkarmamıza yardımcı oluyor ve karmaşıklığı azaltıyor

maxlen=100
x_train=pad_sequences(x_train,maxlen=maxlen,padding='pre',truncating='post',value=0)
y_train=pad_sequences(y_train,maxlen=maxlen,padding='pre',truncating='post',value=0)


"""
Şimdi yukarıda bize x_train , y_train, x_test , y_test içerisinde imdb datasetin
kelimelerinin sayısal hale çevrilmiş şekilleri sunulmaktadır ancak her cümlenin uzunluğu
aynı değildir sinir ağı modelleri her bir inputun aynı uzunlukta olmasını istemektedir.

pad_sequences bu dizilerin başına (veya sonuna) sıfırlar (0) ekleyerek tüm 
dizileri aynı uzunluğa getirir.

Bu durumda, her bir yorum (sequence), en fazla 100 uzunlukta olacak:
-uzunluğu 100’den küçükse, başına sıfır eklenir (varsayılan).
-Uzunluğu 100’den büyükse, son kısmı kesilir.

padding='pre': sıfırlar başa eklenir (varsayılan). 'post' dersek sona eklenir.
truncating='pre': uzun diziler baştan kesilir. 'post' dersek sondan kesilir.
value=0: doldurulacak sayı (genelde 0 kullanılır).

"""

def build_lstm_model():
    model=Sequential()
    """
    Bu katman, kelime indekslerini (sayısal olarak temsil edilen) vektörlere dönüştürür.
    
    input_dim=max_features: Eğitimde kullanılacak toplam kelime sayısı (örneğin, en sık geçen 10.000 kelime).
    output_dim=64: Her kelime vektörü 64 boyutlu olacak.
    input_length=maxlen: Giriş dizisinin maksimum uzunluğu. 
    Tüm diziler bu uzunlukta olmalı (önceden pad edilerek sabitlenmiş).
    """
    model.add(Embedding(input_dim=max_features,output_dim=64,input_length=maxlen))
    model.add(LSTM(units=10))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation="sigmoid"))
    """
    binary classification yapıyoruz 2 adet sonuç çıktımız var eğer 2 den fazla olursa softmax kullanırız
    """
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    
    """
    optimizer=Adam: Optimizasyon algoritması.
    learning_rate=0.0001: Temkinli öğrenme amacıyla öğrenme hızı düşük tutuldu.
    loss="binary_crossentropy": İkili sınıflandırmada kullanılan standart kayıp fonksiyonu.
    metrics=["accuracy"]: Performans ölçütü olarak doğruluk oranı izlenir.
    """
    return model

model=build_lstm_model()
model.summary() #modelin yapısını ve her katmandaki parametre sayısını özetler.


"""
Eğitim sırasında doğrulama kaybı (val_loss) gelişmiyorsa erken durdurma uygulanır.

monitor="val_loss": İzlenecek metrik doğrulama kaybı.
patience=3: 3 epoch boyunca iyileşme olmazsa durdur.
restore_best_weights=True: En düşük doğrulama kaybına sahip ağırlıkları geri yükle.
verbose=2: Eğitim sırasında neden durdurulduğunu açıkça yazar.

"""
early_stopping=EarlyStopping(monitor="val_loss",patience=3,verbose=2,restore_best_weights=True)


"""
x_train, y_train: Eğitim verisi ve etiketler.
epochs=10: Maksimum 10 tur eğitim yapılacak early stopping varsa daha erken bitebilir.
batch_size=16: Her adımda 16 örnek işlenecek.
validation_split=0.2: Eğitim verisinin %20’si doğrulama için ayrılır.
callbacks=[early_stopping]: Early stopping özelliği aktif edilir.
"""

history=model.fit(x_train,y_train,
          epochs=10,
          batch_size=16,
          validation_split=0.2,
          callbacks=[early_stopping]
          )

#model test için evaluate parametresini kullandık bu bize accuracy ve loss değerlerini return eder
accuracy,loss=model.evaluate(x_test,y_test)



import matplotlib.pyplot as plt
plt.figure()

#loss
plt.subplot(1,2,1) #1 satır ve 2 sütundan oluşan bir grafik düzeni oluşturuluyor. Bu ilk grafik olacak.
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
#Eğitim ve doğrulama sırasında her epoch'taki kayıplar çizilir.
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)


#accuracy
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)



























