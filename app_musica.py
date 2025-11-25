import streamlit as st
import librosa
import numpy as np
import tempfile
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Detector de Acordes IA", page_icon="🎵")

st.title("🎵 Transcritor de Áudio para Cifras (Protótipo)")
st.write("Faça upload da sua música do SUNO para detectar o Tom e os Acordes.")

# --- Funções de Análise Musical ---

def estimate_key(chroma):
    """
    Estima o tom global comparando com perfis de Major/Minor (Krumhansl-Schmuckler)
    """
    # Perfis teóricos para acordes Maiores e Menores
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalizar perfis
    major_profile /= np.linalg.norm(major_profile)
    minor_profile /= np.linalg.norm(minor_profile)
    
    # Calcular a média do croma da música inteira
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean /= np.linalg.norm(chroma_mean)
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    max_corr = -1
    best_key = ""
    
    # Correlacionar para cada uma das 12 notas
    for i in range(12):
        # Rotacionar perfil para testar cada tônica
        profile_maj = np.roll(major_profile, i)
        profile_min = np.roll(minor_profile, i)
        
        corr_maj = np.dot(chroma_mean, profile_maj)
        corr_min = np.dot(chroma_mean, profile_min)
        
        if corr_maj > max_corr:
            max_corr = corr_maj
            best_key = f"{notes[i]} Maior"
            
        if corr_min > max_corr:
            max_corr = corr_min
            best_key = f"{notes[i]} Menor"
            
    return best_key

def chords_from_chroma(chroma, sr, hop_length):
    """
    Identifica acordes quadro a quadro simplificado
    """
    # Definição simplificada de templates de acordes (Tríades)
    # 12 notas x 12 tons (Maior) + 12 tons (Menor) = 24 templates
    templates = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        # Maior: Tônica, Terça Maior (+4), Quinta Justa (+7)
        vec = np.zeros(12)
        vec[i] = 1
        vec[(i+4)%12] = 1
        vec[(i+7)%12] = 1
        templates[f"{root}"] = vec # Ex: C
        
        # Menor: Tônica, Terça Menor (+3), Quinta Justa (+7)
        vec_m = np.zeros(12)
        vec_m[i] = 1
        vec_m[(i+3)%12] = 1
        vec_m[(i+7)%12] = 1
        templates[f"{root}m"] = vec_m # Ex: Cm

    detected_chords = []
    times = []
    
    # Transpor chroma para iterar pelo tempo
    chroma_t = chroma.T 
    
    # Processar a cada X frames para não ficar muito denso (suavização)
    frames_per_chord = 30 
    
    for t in range(0, chroma_t.shape[0], frames_per_chord):
        segment = chroma_t[t:t+frames_per_chord]
        if segment.shape[0] == 0: break
        
        # Média do segmento
        avg_vec = np.mean(segment, axis=0)
        
        best_score = -1
        best_chord = "N.C." # No Chord
        
        # Se a energia for muito baixa, é silêncio
        if np.sum(avg_vec) > 0.1:
            for name, template in templates.items():
                score = np.dot(avg_vec, template)
                if score > best_score:
                    best_score = score
                    best_chord = name
        
        timestamp = librosa.frames_to_time(t, sr=sr, hop_length=hop_length)
        
        # Só adiciona se mudou o acorde ou é o primeiro
        if not detected_chords or detected_chords[-1]['chord'] != best_chord:
            detected_chords.append({'time': timestamp, 'chord': best_chord})

    return detected_chords

# --- Interface do Usuário ---

uploaded_file = st.file_uploader("Escolha um arquivo de áudio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    with st.spinner('A IA está ouvindo e analisando...'):
        # Salvar arquivo temporário para o Librosa ler
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 1. Carregar Áudio
            y, sr = librosa.load(tmp_path, sr=11025, duration=60)
            
            # 2. Separar Harmonia (melhora detecção de acordes)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 3. Extrair Chroma (Mapa de calor das 12 notas)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            
            # 4. Detectar Tom Global
            key = estimate_key(chroma)
            st.success(f"🔑 Tonalidade Detectada: **{key}**")
            
            # 5. Detectar Sequência de Acordes
            st.subheader("📜 Sequência de Acordes")
            chords = chords_from_chroma(chroma, sr, 512)
            
            # Formatar para exibição
            chord_str = ""
            for item in chords:
                # Exibe tempo e acorde
                time_str = f"{int(item['time'] // 60)}:{int(item['time'] % 60):02d}"
                chord_str += f"**[{time_str}]** {item['chord']}  ➡️  "
            
            st.markdown(chord_str)
            
            # Visualização Gráfica (Opcional - Cromagrama)
            st.subheader("📊 Visualização das Notas (Chromagram)")
            st.pyplot(librosa.display.specshow(chroma, y_axis='chroma', x_axis='time').figure)

        except Exception as e:
            st.error(f"Erro ao processar: {e}")
        finally:

            os.remove(tmp_path) # Limpeza

