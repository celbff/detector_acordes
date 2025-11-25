import streamlit as st
import librosa
import librosa.display
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(page_title="Detector de Acordes IA", page_icon="🎵")

st.title("🎵 Transcritor de Áudio para Cifras (Protótipo)")
st.write("Faça upload da sua música do SUNO para detectar o Tom e os Acordes.")

# --- Dicionário de Diagramas de Acordes (Representação em ASCII/Texto) ---
# Formato: [Acorde] -> [Diagrama de 6 cordas (E A D G B e)]
# X = Não tocar (mute), 0 = Corda Solta, 1-6 = Casa a pressionar
GUITAR_CHORD_FINGERINGS = {
    "C": """
   C
e|-0-|
B|-1-|
G|-0-|
D|-2-|
A|-3-|
E|---|
""",
    "Cm": """
   Cm
e|-3-|
B|-4-|
G|-5-|
D|-5-|
A|-3-|
E|---|
""",
    "G": """
   G
e|-3-|
B|-0-|
G|-0-|
D|-0-|
A|-2-|
E|-3-|
""",
    "Gm": """
   Gm
e|-3-|
B|-3-|
G|-3-|
D|-5-|
A|-5-|
E|-3-|
""",
    "D": """
   D
e|-2-|
B|-3-|
G|-2-|
D|-0-|
A|---|
E|---|
""",
    "Dm": """
   Dm
e|-1-|
B|-3-|
G|-2-|
D|-0-|
A|---|
E|---|
""",
    "A": """
   A
e|-0-|
B|-2-|
G|-2-|
D|-2-|
A|-0-|
E|---|
""",
    "Am": """
   Am
e|-0-|
B|-1-|
G|-2-|
D|-2-|
A|-0-|
E|---|
""",
    "E": """
   E
e|-0-|
B|-0-|
G|-1-|
D|-2-|
A|-2-|
E|-0-|
""",
    "Em": """
   Em
e|-0-|
B|-0-|
G|-0-|
D|-2-|
A|-2-|
E|-0-|
""",
    "F": """
   F
e|-1-|
B|-1-|
G|-2-|
D|-3-|
A|-3-|
E|-1-|
""",
    "Fm": """
   Fm
e|-1-|
B|-1-|
G|-1-|
D|-3-|
A|-3-|
E|-1-|
""",
    # Acordes com sustenidos e bemóis (F#, G#, A#, C#, D#)
    "F#": """
   F#
e|-2-|
B|-2-|
G|-3-|
D|-4-|
A|-4-|
E|-2-|
""",
    "F#m": """
   F#m
e|-2-|
B|-2-|
G|-2-|
D|-4-|
A|-4-|
E|-2-|
""",
    "G#": """
   G#
e|-4-|
B|-4-|
G|-5-|
D|-6-|
A|-6-|
E|-4-|
""",
    "G#m": """
   G#m
e|-4-|
B|-4-|
G|-4-|
D|-6-|
A|-6-|
E|-4-|
""",
    "A#": """
   A#
e|-6-|
B|-6-|
G|-7-|
D|-8-|
A|-8-|
E|-6-|
""",
    "A#m": """
   A#m
e|-6-|
B|-6-|
G|-6-|
D|-8-|
A|-8-|
E|-6-|
""",
    "C#": """
   C#
e|-4-|
B|-6-|
G|-6-|
D|-6-|
A|-4-|
E|---|
""",
    "C#m": """
   C#m
e|-4-|
B|-5-|
G|-6-|
D|-6-|
A|-4-|
E|---|
""",
    "D#": """
   D#
e|-6-|
B|-8-|
G|-8-|
D|-8-|
A|-6-|
E|---|
""",
    "D#m": """
   D#m
e|-6-|
B|-7-|
G|-8-|
D|-8-|
A|-6-|
E|---|
""",
    "N.C.": "   N.C.\n(Sem Acorde)"
}


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

def detect_beats_and_chords(y_harmonic, sr, chroma):
    """
    Detecta BPM, tempos fortes (beats) e alinha a detecção de acordes com esses tempos.
    """
    # 1. Detecção de Ritmo (Tempo)
    tempo, beats = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    
    # 2. Definição simplificada de templates de acordes (Tríades)
    templates = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        # Maior: Tônica, Terça Maior (+4), Quinta Justa (+7)
        vec_maj = np.zeros(12); vec_maj[i] = 1; vec_maj[(i+4)%12] = 1; vec_maj[(i+7)%12] = 1
        templates[f"{root}"] = vec_maj
        
        # Menor: Tônica, Terça Menor (+3), Quinta Justa (+7)
        vec_min = np.zeros(12); vec_min[i] = 1; vec_min[(i+3)%12] = 1; vec_min[(i+7)%12] = 1
        templates[f"{root}m"] = vec_min
        
    detected_chords = []
    
    # 3. Alinhamento dos Acordes com os Tempos Fortes (Beats)
    for i, beat_frame in enumerate(beats):
        
        # Usa o quadro de chroma que corresponde à batida
        frame_index = beat_frame 
        
        # Garantir que o índice não exceda o tamanho da matriz chroma
        if frame_index >= chroma.shape[1]:
            break 
            
        # Pega o vetor de chroma no momento exato da batida
        avg_vec = chroma[:, frame_index] 
        
        best_score = -1
        best_chord = "N.C." # No Chord
        
        # 4. Correlacionar com os templates
        if np.sum(avg_vec) > 0.1: # Ignora se for muito silencioso
            for name, template in templates.items():
                # Calcula a correlação (produto escalar)
                score = np.dot(avg_vec, template)
                if score > best_score:
                    best_score = score
                    best_chord = name
        
        # Só adiciona se mudou o acorde ou é o primeiro
        if not detected_chords or detected_chords[-1]['chord'] != best_chord:
            detected_chords.append({'beat': i + 1, 'chord': best_chord}) # Batida começa em 1

    return detected_chords, tempo

def display_chord_diagrams(chords_list):
    """
    Exibe os diagramas em ASCII dos acordes únicos encontrados na música.
    """
    # 1. Obter a lista de acordes únicos (e válidos)
    unique_chords = sorted(list(set(item['chord'] for item in chords_list)))
    
    # 2. Filtrar apenas acordes que têm um diagrama
    diagram_chords = [c for c in unique_chords if c in GUITAR_CHORD_FINGERINGS]
    
    if not diagram_chords:
        st.warning("Não foi possível gerar diagramas de acordes para as cifras encontradas.")
        return

    st.subheader("🎸 Diagramas de Acordes para Violão")
    
    # Divide os diagramas em colunas para melhor visualização
    cols = st.columns(min(len(diagram_chords), 4)) 
    
    for i, chord in enumerate(diagram_chords):
        diagram = GUITAR_CHORD_FINGERINGS[chord]
        
        # Usa um bloco de código Markdown para formatar o diagrama em ASCII (Monospace)
        cols[i % 4].markdown(f"```text\n{diagram}\n```")


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
            # 1. Carregar Áudio (Otimizado para o Render)
            y, sr = librosa.load(tmp_path, sr=11025, duration=60)
            
            # 2. Separar Harmonia (melhora detecção de acordes)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 3. Extrair Chroma (Mapa de calor das 12 notas)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            
            # 4. Detectar Tom Global
            key = estimate_key(chroma)
            st.success(f"🔑 Tonalidade Detectada: **{key}**")
            
            # 5. Detectar Sequência de Acordes e Batidas
            st.subheader("📜 Sequência de Acordes (Alinhada por Batida)")
            chords_by_beat, tempo = detect_beats_and_chords(y_harmonic, sr, chroma)
            
            st.info(f"Metrônomo Detectado: **{int(tempo)} BPM**")

            # Formatar e exibir a sequência
            chord_str = ""
            for item in chords_by_beat:
                # Exibe a batida e o acorde
                chord_str += f"**[B:{item['beat']:02d}]** {item['chord']}  ➡️  "
            
            st.markdown(chord_str)
            
            # --- NOVO: Exibir Diagramas de Acordes ---
            display_chord_diagrams(chords_by_beat)
            
            st.markdown("---")
            
            # Visualização Gráfica
            st.subheader("📊 Visualização das Notas (Chromagram)")
            # Usar plt.subplots para garantir que o Streamlit exiba corretamente
            fig, ax = plt.subplots(figsize=(10, 5))
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
            ax.set(title='Chromagram')
            st.pyplot(fig) 
            
            st.markdown("---")
            st.markdown(f"**Próximo Passo:** Use a Batida (`B:xx`) para alinhar a letra. Cada número representa um pulso forte da música. Por exemplo: `[B:01] Amor [B:05] é algo...`")

        except Exception as e:
            st.error(f"Erro ao processar: {e}. (Verifique se o arquivo de áudio é válido.)")
        finally:
            # Limpeza do arquivo temporário
            if os.path.exists(tmp_path):
                 os.remove(tmp_path)
