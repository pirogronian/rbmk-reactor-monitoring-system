# sanity_check.py
import sys
from pathlib import Path
import torch

# 1. Dodajemy folder 'src' do ścieżki systemowej, dokładnie tak jak w main.py
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importujemy Twój loader z pliku dataset.py
from dataset import RBMKDataLoader

def run_test():
    print("⏳ [SANITY CHECK] Uruchamiam test potoku danych...")
    
    # Ścieżka do Twojego pliku z danymi (dopasuj, jeśli masz inną)
    DATA_PATH = "data/Influx_RBML_data.parquet"
    
    try:
        # 2. Inicjalizujemy loader z takimi samymi parametrami jak w configu
        print("-> Inicjalizacja RBMKDataLoader...")
        data_loader = RBMKDataLoader(
            data_path=DATA_PATH,
            sequence_length=50,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        # 3. Próbujemy odpalić cały proces: load -> split -> scale -> dataloader
        print("-> Próba utworzenia PyTorch DataLoaders (tu sprawdzamy poprawność skalowania)...")
        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            batch_size=32,
            num_workers=0  # Na czas testu dajemy 0, żeby łatwiej wyłapać błędy
        )
        
        # 4. KLUCZ PROGRAMU: Pobieramy dokładnie JEDNĄ paczkę danych (batch)
        # iter() zamienia loader w iterator, a next() wyciąga z niego pierwszą paczkę
        print("-> Próba pobrania pierwszego batcha danych z train_loader...")
        first_batch = next(iter(train_loader))
        
        # Sprawdzamy, czy loader zwraca krotkę (wejście, cel) czy pojedynczy obiekt,
        # co zależy od tego, czy poprawiłeś/aś krok z nadmiarowością danych
        if isinstance(first_batch, (list, tuple)):
            sequences = first_batch[0]
            print("   [Info] Loader zwraca parę (wejście, target).")
        else:
            sequences = first_batch
            print("   [Info] Loader zwraca pojedynczy tensor.")
            
        # 5. Wyświetlamy raport z sukcesu i wymiary tensorów
        print("\n" + "="*50)
        print("✅ SANITY CHECK ZAKOŃCZONY SUKCESEM!")
        print("="*50)
        print(f"📊 Kształt tensora (Shape): {sequences.shape}")
        print(f"   • Rozmiar paczki (Batch Size): {sequences.shape[0]} (oczekiwano: 32)")
        print(f"   • Długość okna (Seq Length):  {sequences.shape[1]} (oczekiwano: 50)")
        print(f"   • Liczba cech reaktora:       {sequences.shape[2]}")
        print("="*50 + "\n")
        
    except Exception as e:
        # Jeśli gdzieś pojawiła się literówka lub błąd logiczny - tu go zobaczysz
        print("\n" + "!"*50)
        print("❌ SANITY CHECK NIE POWIÓDŁ SIĘ!")
        print("!"*50)
        print(f"Treść błędu: {e}")
        print("!"*50 + "\n")

if __name__ == "__main__":
    run_test()