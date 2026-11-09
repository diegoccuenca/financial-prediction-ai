from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# CSV-bestand inlezen
df = pd.read_csv("data/omzet.csv")

# Eerste vijf rijen tonen
print(df.head())

# Gemiddelde omzet berekenen
gemiddelde = df["omzet"].mean()

# hoogste waard vinden 
hoogste_waarde = df["omzet"].max()

# laagste waarde vinden 
laagste_waarde = df["omzet"].min()

# totale omzet berekenen
totale_omzet = df["omzet"].sum()


print(f"Gemiddelde omzet: €{gemiddelde:.2f}")
print(f"hoogste waarde: €{hoogste_waarde:.2f}")
print(f"laagste waarde: €{laagste_waarde:.2f}")
print(f"totale omzet: €{totale_omzet:.2f}")

##############################################################################

# Verband tussen advertentiebudget en omzet
correlatie = df["advertentiebudget"].corr(df["omzet"])

print(f"Correlatie: {correlatie:.2f}")

# correlatie tussen websitebezoekers en omzet
correlatie_bezoekers = df["bezoekers"].corr(df["omzet"])

print(f"correlatie tussen bezoekers en omzet: {correlatie_bezoekers:.2f}")

#######################################################################################

# Grafiek maken
plt.scatter(df["advertentiebudget"], df["omzet"])

# Titels toevoegen
plt.title("Advertentiebudget vs Omzet")
plt.xlabel("Advertentiebudget (€)")
plt.ylabel("Omzet (€)")

# Grafiek weergeven
plt.show()

#grafiek maken
plt.scatter(df["bezoekers"], df["omzet"])

#titels toevoegen
plt.title("bezoekers vs omzet")
plt.xlabel("bezoekers (aantal)")
plt.ylabel("omzet (euros)")

#grafiek weergeven
plt.show()

###########################################################################################

# Input en target bepalen
X = df[["advertentiebudget"]]
y = df["omzet"]

# AI-model aanmaken
model = LinearRegression()

# Model trainen
model.fit(X, y)

# Nieuwe gegevens
nieuw_budget = pd.DataFrame({
    "advertentiebudget": [4000]
})

# Voorspelling maken
voorspelling = model.predict(nieuw_budget)

print(f"Voorspelde omzet: €{voorspelling[0]:.2f}")

################################################################################################



# Voorspellingen maken voor de bestaande dataset
voorspellingen = model.predict(X)

# Gemiddelde fout berekenen
fout = mean_absolute_error(y, voorspellingen)

print(f"Gemiddelde voorspelfout: €{fout:.2f}")

#####################################################################################################\\



# Data verdelen: 75% trainen, 25% testen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

# Nieuw model maken
test_model = LinearRegression()

# Alleen trainen met trainingsdata
test_model.fit(X_train, y_train)

# Voorspellen met testdata
test_voorspellingen = test_model.predict(X_test)

# Voorspelfout berekenen
test_fout = mean_absolute_error(y_test, test_voorspellingen)

print(f"Testfout: €{test_fout:.2f}")

#############################################################################################

# Voorspellingen op trainingsdata
train_voorspellingen = test_model.predict(X_train)

# Trainingsfout berekenen
train_fout = mean_absolute_error(y_train, train_voorspellingen)

print(f"Trainingsfout: €{train_fout:.2f}")
print(f"Testfout: €{test_fout:.2f}")

verschil = test_fout - train_fout

print(f"Verschil: €{verschil:.2f}")

#####################################################################################################

X_nieuw = df[["advertentiebudget", "bezoekers"]]
y_nieuw = df["omzet"]


X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_nieuw, y_nieuw, test_size=0.25, shuffle=False
)


# Nieuw model maken
test_model2 = LinearRegression()

# Alleen trainen met trainingsdata
test_model2.fit(X_train2, y_train2)

# Voorspellen met testdata
test_voorspellingen2 = test_model2.predict(X_test2)


# Voorspelfout berekenen
test_fout2 = mean_absolute_error(y_test2, test_voorspellingen2)

print(f"Testfout model 1: €{test_fout:.2f}")
print(f"Testfout model 2: €{test_fout2:.2f}")

##########################################################################################