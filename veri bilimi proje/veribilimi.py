import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/yazel/Desktop/veri bilimi proje/brain_stroke.csv")
df.head()
df.dtypes
df.duplicated()
df.isnull()
df.isnull().sum()
# ısı haritasında görelim
sns.set_theme() #seaborn kütüphanesinin önceden tanımlı bir tema kullanmasını sağlar
sns.set(rc={"figure.dpi":300, "figure.figsize":(12,9)})
sns.heatmap(df.isnull(), cbar=False) #cbar=False renk çubuğunun (colorbar) gösterilmemesini sağlar.

# eksik verileri tamamlama
df.ffill(inplace=True)
df.head()
# df.fillna(method = 'ffill', inplace=True)  # ileri doldurma

# yaş sütununda bulunan anlamsız verileri dönüştürme
df.loc[df["age"] == 0, "age"] = np.nan # 0ları nan ile değiştir
age_median = df["age"].median()
df["age"].fillna(age_median)
df.head()

# kadınlarda mı erkeklerde mi beyin felcine daha çok rastlanmıştır
# Cinsiyet ve beyin felci durumu arasındaki ilişkiyi gösteren çapraz tabloyu oluştur
cross_table = pd.crosstab(df['gender'], df['stroke'])
print(cross_table)

# beyin felci geçirenlerin ortalama yaşı nedir
average_age_stroke = df[df['stroke'] == 1]['age'].mean()
print("Beyin felci geçirenlerin ortalama yaşı:", average_age_stroke)

# beyin felci geçirenler çoğunlukla nerede yaşıyor
# Beyin felci geçirenlerin yaşam yerlerini say
stroke_by_residence = df[df['stroke'] == 1]['Residence_type'].value_counts() # moddan farkı her benzersiz değerin sayısını verir. mod birden fazla olabilir
print("Beyin felci geçirenlerin yaşadığı yerler:") 
print(stroke_by_residence)

# Hangi demografik grupta inme riski daha yüksektir
df=pd.read_csv("C:/Users/yazel/Desktop/veri bilimi proje/brain_stroke.csv").head(1000)
demographic_groups = ['gender','work_type', 'Residence_type']
stroke_counts = df.groupby(demographic_groups)['stroke'].sum() # Her bir demografik grup için inme vakası sayılarını hesapla
stroke_counts
total_counts = df.groupby(demographic_groups)['stroke'].count() # Her bir demografik grup için toplam hasta sayılarını hesapla
total_counts
# Her bir demografik gruptaki inme vakası oranlarını hesapla
stroke_rates = stroke_counts / total_counts * 100  # Yüzde cinsinden ifade etmek için 100 ile çarp

# Grafikle görselleştirme
stroke_rates.plot(kind='bar', figsize=(10,6), xlabel='Demografik Grup', ylabel='İnme Vakası Oranı (%)', title='Demografik Gruplara Göre İnme Vakası Oranları')
plt.xticks(rotation=85)  # x eksenindeki etiketleri döndür
plt.tight_layout()  # Grafiği sıkıştır
plt.show()

# hipertansiyon beyin felcini etkiler mi
df=pd.read_csv("C:/Users/yazel/Desktop/veri bilimi proje/brain_stroke.csv")
from scipy.stats import chi2_contingency

# İki değişken arasındaki ilişkiyi görmek/test etmek için bir kontingensi tablosu oluşturun
contingency_table = pd.crosstab(df['hypertension'], df['stroke'])
contingency_table
# Ki-kare testi yap
chi2, p_value = chi2_contingency(contingency_table)[:2]
p_value
# P değerini kontrol ederek hipotezi değerlendirin
alpha = 0.05
if p_value < alpha:
    print("H0 hipotezi reddedilir. Hipertansiyon beyin felci riskini artırır.")
else:
    print("H0 hipotezi kabul edilir. Hipertansiyonun beyin felci üzerinde etkisi yoktur.")

# kalp hastalığı beyin felcini etkiler mi
df=pd.read_csv("C:/Users/yazel/Desktop/veri bilimi proje/brain_stroke.csv").tail(30)
from scipy.stats import ttest_ind

# Heart disease olan ve olmayan grupları ayır
heart_disease_yes = df[df['heart_disease'] == 1]['stroke']
heart_disease_no = df[df['heart_disease'] == 0]['stroke']

# T-testi yap
t_stat, p_val = ttest_ind(heart_disease_yes, heart_disease_no)

# p değerini kontrol et ve hipotezi değerlendir
alpha = 0.05
if p_val < alpha:
    print("H0 hipotezi reddedilir. Kalp hastalığı ile inme arasında anlamlı bir ilişki vardır.")
else:
    print("H0 hipotezi kabul edilir. Kalp hastalığı ile inme arasında anlamlı bir ilişki yoktur.")

# 'bmi' değişkeninin histogramını çiz (normal dağılıma sahip olduğu için)
df=pd.read_csv("C:/Users/yazel/Desktop/veri bilimi proje/brain_stroke.csv")
plt.hist(df['bmi'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Histogram of BMI')
plt.show()

# BMI'nin 30'dan yüksek olma olasılığı nedir? ortalama ve standart sapma biliniyor
bmi_mean = df['bmi'].mean()
print("BMI'nin ortalaması:", bmi_mean)
bmi_std = df['bmi'].std()
print("BMI'nin standart sapması:", bmi_std)
from scipy.stats import norm 
1-norm.cdf(30,28,6)


# bu veri setinde anakütleden alınan bir örneklemin ortalama gkukoz seviyesinin anakütle ortalamasından yüksek olduğu iddia edilmektedir.
#100 kişinin ortalama glukoz seviyelerinin ortalaması hesaplanmış olup 112 olarak bulunmuştur.
#buna göre anakütlenin ortalama değeri 105 ve standart sapma 48 olduğuna göre verilen iddianın %5 anlamlılık düzeyinde doğru olup olmadığını belirtin

from scipy.stats import norm

# Verilen veriler
n = 100  # Örneklemin büyüklüğü
x = 112  # Örneklemin ortalama glukoz seviyesi
m = 105  # Anakütle ortalaması
sigma = 48  # Anakütle standart sapması

# Z-testi istatistiğinin hesaplanması
z = (x - m ) / (sigma / np.sqrt(n))

# %5 anlamlılık düzeyi için sağa tek yönlü kritik değeri hesapla
alpha = 0.05
z_critical = norm.ppf(1 - alpha)

# Karar
if z > z_critical:
    print("H0 reddedilir: Örneklemin ortalama glukoz seviyesi anakütle ortalamasından yüksektir.")
else:
    print("H0 reddedilmez: Örneklemin ortalama glukoz seviyesi anakütle ortalamasından yüksek değildir.")


#evli olan kişilerde evli olmayanlara göre daha çok inmeye rastlandığı hipotezi ortaya atılıyor doğru olup olmadığına bakalım
from scipy.stats import norm

# Veriler
married_stroke = df[(df['ever_married'] == 'Yes') & (df['stroke'] == 1)]
unmarried_stroke = df[(df['ever_married'] == 'No') & (df['stroke'] == 1)]

# Her iki gruptaki sadece inme geçiren toplam hasta sayısı
n_married = len(married_stroke)
n_unmarried = len(unmarried_stroke)

# Her iki gruptaki toplam hasta sayısı
n_total_married = len(df[df['ever_married'] == 'Yes'])
n_total_unmarried = len(df[df['ever_married'] == 'No'])

# Her iki grupta inme geçirenlerin oranı
p_married = n_married / n_total_married
p_unmarried = n_unmarried / n_total_unmarried

# Z-testi istatistiği hesaplama
z = (p_married - p_unmarried) / ((p_married * (1 - p_married) / n_total_married + p_unmarried * (1 - p_unmarried) / n_total_unmarried) ** 0.5)

# %5 anlamlılık düzeyi için kritik değerler
alpha = 0.05
z_critical = norm.ppf(1 - alpha / 2)  # İki yönlü test olduğu için 2'ye bölünür.

# Karar
if z > z_critical or z < -z_critical:
    print("H0 reddedilir: Evli olanlar ile evli olmayanlar arasında inme riski açısından bir fark vardır.")
else:
    print("H0 reddedilmez: Evli olanlar ile evli olmayanlar arasında inme riski açısından bir fark yoktur.")


# yaş ortalaması güven aralığı 
#popülasyonun gerçek yaş ortalamasının muhtemel bir aralığını verir, bu da örneklemin ne kadar güvenilir bir şekilde popülasyonu temsil ettiğini belirlemeye yardımcı olur.

from scipy.stats import t

# 'age' değişkeninden örnek al
age_sample = df['age']

# Örneklem büyüklüğü
n = len(age_sample)

# Ortalama ve standart sapma
age_mean = np.mean(age_sample)
age_std = np.std(age_sample, ddof=1)

# Güven düzeyi ve güven aralığı hesapla
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = t.ppf(1 - alpha / 2, df=n - 1)
margin_of_error = t_critical * age_std / np.sqrt(n)
confidence_interval = (age_mean - margin_of_error, age_mean + margin_of_error)

print("Güven aralığı (95% güven düzeyi) yaş:", confidence_interval)

# normalizasyon

from sklearn.preprocessing import MinMaxScaler

# Min-Max normalizasyonu için MinMaxScaler'ı oluştur
scaler = MinMaxScaler()

# 'avg_glucose_level' ve 'bmi' değişkenlerini seç ve normalizasyon yap
normalized_data = df[['avg_glucose_level', 'bmi']]
normalized_data = scaler.fit_transform(normalized_data)

# Normalizasyon sonrası veriyi DataFrame'e dönüştür
normalized_df = pd.DataFrame(normalized_data, columns=['normalized_avg_glucose_level', 'normalized_bmi'])

# İlk 5 satırı göster
print(normalized_df.head())
