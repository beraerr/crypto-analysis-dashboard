@echo off
echo ========================================
echo Kripto Tahmin Uygulamasi Baslatiliyor...
echo ========================================
echo.

REM Gerekli kutuphaneleri kontrol et ve yukle
echo Gerekli kutuphaneler kontrol ediliyor...
pip install -r requirements_crypto.txt

echo.
echo Uygulama baslatiliyor...
echo Tarayicinizda otomatik olarak acilacaktir.
echo.

streamlit run crypto_prediction_app.py

pause

