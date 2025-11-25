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
st.write("Faça upload da sua música do SUNO e da letra para análise e sincronização.")

# --- Dicionários de Recursos Musicais ---

# Dicionário de digitações de acordes (Mantido o mesmo)
GUITAR_CHORD_FINGERINGS = {
    # [Dicionário de Acordes: C, Cm, G, Gm, etc.]
    # ... (Conteúdo original de GUITAR_CHORD_FINGERINGS) ...
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


# Tonalidade -> Intervalos da Escala Pentatônica (em semitons)
SCALE_INTERVALS = {
    "Maior": [0, 2, 4, 7, 9],
    "Menor": [0, 3, 5, 7, 10]
}

NOTES_DICT = {i: note for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}

# --- Funções de Análise Musical ---

def estimate_key(chroma):
    # [Função de estimate_key mantida]
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    major_profile /= np.linalg.norm(major_profile)
    minor_profile /= np.linalg.norm(minor_profile)
    
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean /= np.linalg.norm(chroma_mean)
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    max_corr = -1
    best_key = ""
    
    for i in range(12):
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

def filter_short_chord_changes(chords_list):
    """
    Filtra acordes que duram apenas um beat e que retornam
    imediatamente ao acorde anterior (evita 'blips' de detecção: A -> B -> A).
    Isso ajuda a simplificar a transcrição de acordes.
    """
    if len(chords_list) < 3:
        return chords_list

    filtered_list = [chords_list[0]]
    i = 1
    while i < len(chords_list) - 1:
        current_item = chords_list[i]
        prev_item = filtered_list[-1]
        next_item = chords_list[i+1]
        
        # Verifica o padrão: Acorde A -> Acorde B (1 beat) -> Acorde A
        # O acorde 'current_item' é um 'blip' se ele é diferente do anterior,
        # e o próximo acorde é igual ao anterior.
        
        # O teste é se o acorde atual dura apenas um beat *e* se o acorde vizinho é igual ao anterior.
        # Devido à lógica de detecção, o acorde "current" já tem o beat imediatamente 
        # após o beat do "prev_item"
        if prev_item['chord'] == next_item['chord'] and \
           prev_item['chord'] != current_item['chord']:
           
            # Pula o acorde 'current_item' (o blip) e continua a iteração, 
            # mantendo o acorde anterior (prev_item) no filtered_list
            i += 1
            continue

        filtered_list.append(current_item)
        i += 1
        
    # Adiciona o último acorde, se não foi processado
    if i == len(chords_list) - 1:
        filtered_list.append(chords_list[-1])

    # Caso a lista tenha sido reduzida a 1 elemento (por ser curta demais), retorna ela.
    if not filtered_list:
        return chords_list[:1] 

    return filtered_list


def detect_beats_and_chords(y_harmonic, sr, chroma):
    # [Função de detect_beats_and_chords mantida]
    tempo, beats = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    
    templates = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        vec_maj = np.zeros(12); vec_maj[i] = 1; vec_maj[(i+4)%12] = 1; vec_maj[(i+7)%12] = 1
        templates[f"{root}"] = vec_maj
        
        vec_min = np.zeros(12); vec_min[i] = 1; vec_min[(i+3)%12] = 1; vec_min[(i+7)%12] = 1
        templates[f"{root}m"] = vec_min
        
    detected_chords = []
    beat_frames = [] 
    for i, beat_frame in enumerate(beats):
        
        frame_index = beat_frame 
        beat_frames.append(frame_index) 
        
        if frame_index >= chroma.shape[1]:
            break 
            
        avg_vec = chroma[:, frame_index] 
        
        best_score = -1
        best_chord = "N.C." 
        
        if np.sum(avg_vec) > 0.1: 
            for name, template in templates.items():
                score = np.dot(avg_vec, template)
                if score > best_score:
                    best_score = score
                    best_chord = name
        
        # Só adiciona se for um acorde diferente do último
        if not detected_chords or detected_chords[-1]['chord'] != best_chord:
            # Evita acumular N.C.s consecutivos
            if best_chord == "N.C." and detected_chords and detected_chords[-1]['chord'] == "N.C.":
                continue
            detected_chords.append({'beat': i + 1, 'chord': best_chord, 'frame': frame_index})

    # NOVO: Aplica a filtragem para remover mudanças de acordes muito curtas (blips)
    final_chords = filter_short_chord_changes(detected_chords)

    return final_chords, tempo, beat_frames 

def display_chord_diagrams(chords_list):
    """
    Exibe os diagramas em ASCII dos acordes únicos encontrados na música.
    Inclui um log de quais acordes foram detectados para fins de depuração.
    """
    # 1. Extrai os acordes únicos detectados
    unique_chords = sorted(list(set(item['chord'] for item in chords_list)))
    
    # Adiciona log de debug para o usuário
    st.code(f"Acordes Únicos Detectados: {unique_chords}", language="text")

    # 2. Filtra a lista para apenas os acordes que temos diagramas
    diagram_chords = [c for c in unique_chords if c in GUITAR_CHORD_FINGERINGS]
    
    if not diagram_chords:
        st.warning("Não foi possível gerar diagramas de acordes para as cifras encontradas.")
        return

    st.subheader("🎸 Diagramas de Acordes para Violão")
    
    cols = st.columns(min(len(diagram_chords), 4)) 
    
    for i, chord in enumerate(diagram_chords):
        diagram = GUITAR_CHORD_FINGERINGS[chord]
        cols[i % 4].markdown(f"```text\n{diagram}\n```")


def display_scale_suggestion(key_name):
    """
    Sugere e exibe a escala pentatônica para a tonalidade detectada.
    """
    # [Função de display_scale_suggestion mantida]
    if ' ' not in key_name:
        return 
        
    root, quality = key_name.split() 
    
    scale_type = 'Maior' if quality == 'Maior' else 'Menor'
    intervals = SCALE_INTERVALS.get(scale_type, [])
    
    if not intervals:
        return
        
    root_index = [i for i, note in NOTES_DICT.items() if note == root]
    if not root_index:
        return
    root_index = root_index[0]
    
    scale_notes = sorted([NOTES_DICT[(root_index + interval) % 12] for interval in intervals])
    scale_name = f"{root} {scale_type} Pentatônica"

    st.subheader("🎼 Sugestão para Riffs e Introduções")
    st.markdown(f"A melhor escala para solos e melodias é a **{scale_name}**.")
    st.markdown(f"**Notas:** {', '.join(scale_notes)}")
    
    diagram = f"""
   Escala: {scale_name}
(T = Tônica, • = Nota da Escala)

e|-----------------•---T---|  <-- Casa 17-20
B|---------------•---T---•-|  <-- Casa 15-18
G|-------------•---T---•---|  <-- Casa 14-17
D|-----------•---•---T-----|  <-- Casa 12-15
A|---------•---•---T-------|  <-- Casa 12-15
E|-------•---T---•---------|  <-- Casa 12-15

"""
    st.markdown("Use esta **'Caixa de Escala'** para tocar riffs.")
    st.code(diagram, language='text')
    
    
def format_and_display_chords(chords_list, beats_per_line=4):
    """
    Formata a sequência de acordes em linhas, simulando compassos.
    """
    st.info("A sequência abaixo mostra a cifra no momento em que a batida muda. Um acorde é mantido até a próxima cifra aparecer.")
    
    markdown_output = "```markdown\n"
    
    beat_counter = 1 
    
    for item in chords_list:
        chord = item['chord']
        
        # Adiciona o acorde com a tag de batida
        beat_tag = f"[B:{item['beat']:02d}]"
        markdown_output += f"{beat_tag} **{chord}** "
        
        # Quebra de linha a cada 'beats_per_line'
        if beat_counter % beats_per_line == 0:
            markdown_output += "\n"
        else:
            markdown_output += " | " # Separador visual para o compasso
        
        beat_counter += 1
        
    markdown_output += "\n```"
    st.markdown(markdown_output)


def display_lyrics_with_chords(lyrics_content, chords_list):
    """
    Tenta sincronizar os acordes detectados com as linhas da letra,
    fazendo uma distribuição básica.
    """
    
    # 1. Limpa e divide a letra em linhas significativas
    lines = [line.strip() for line in lyrics_content.split('\n') if line.strip()]
    num_lines = len(lines)
    num_chords = len(chords_list)
    
    if num_lines == 0 or num_chords == 0:
        st.warning("Letra ou acordes não disponíveis para sincronização.")
        return
        
    st.info(f"O Canvas tentou distribuir {num_chords} acordes por {num_lines} linhas de letra. Ajuste a posição dos acordes na caixa de texto abaixo.")

    # 2. Distribui os acordes de forma básica
    # Calcula quantos acordes em média por linha e o resto
    chords_per_line = num_chords // num_lines
    remainder = num_chords % num_lines
    
    output = ""
    chord_index = 0
    
    for i, line in enumerate(lines):
        # Determina quantos acordes esta linha receberá
        current_line_chord_count = chords_per_line + (1 if i < remainder else 0)
        
        chords_for_line = []
        for j in range(current_line_chord_count):
            if chord_index < num_chords:
                chords_for_line.append(chords_list[chord_index]['chord'])
                chord_index += 1
            else:
                break
        
        chord_line = ""
        if chords_for_line:
            # Junta os acordes com 4 espaços como separador para alinhamento
            chord_line = "    ".join(chords_for_line)
        
        # Formato de saída: Linha de acordes (pode estar vazia) + Linha da letra
        output += f"{chord_line}\n{line}\n\n"
        
    # 3. Exibe o resultado para edição manual do usuário
    st.text_area(
        "Edite e Sincronize a Letra (Acordes Estimados Acima da Linha)", 
        value=output.strip(), 
        height=400,
        help="Os acordes foram distribuídos automaticamente, mas o alinhamento de tempo (sincronização) é aproximado. Você pode mover os acordes para o ponto exato onde a letra/música os requer."
    )


# --- Interface do Usuário ---

# Organizar Uploads em colunas para melhor visualização
col_audio, col_lyrics = st.columns(2)

with col_audio:
    uploaded_audio = st.file_uploader("1. Escolha o Áudio (MP3/WAV)", type=["mp3", "wav"])

with col_lyrics:
    uploaded_lyrics = st.file_uploader("2. Escolha a Letra (TXT)", type=["txt"])


if uploaded_audio is not None:
    st.audio(uploaded_audio, format='audio/mp3')
    
    with st.spinner('A IA está ouvindo e analisando...'):
        # Salvar arquivo temporário para o Librosa ler
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_audio.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 1. Carregar Áudio (REVERTIDO para 60 segundos para evitar erro de memória/502)
            y, sr = librosa.load(tmp_path, sr=11025, duration=60)
            
            # 2. Separar Harmonia (melhora detecção de acordes)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 3. Extrair Chroma (Mapa de calor das 12 notas)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            
            # 4. Detectar Tom Global
            key = estimate_key(chroma)
            st.success(f"🔑 Tonalidade Detectada: **{key}**")
            
            # --- Sugestão de Escala para Tablatura/Riffs ---
            display_scale_suggestion(key)
            
            # 5. Detectar Sequência de Acordes e Batidas
            st.subheader("📜 Sequência de Acordes (Alinhada por Batida)")
            chords_by_beat, tempo, beat_frames = detect_beats_and_chords(y_harmonic, sr, chroma)
            
            st.info(f"Metrônomo Detectado: **{int(tempo)} BPM**")

            # --- Formatação de Acordes ---
            format_and_display_chords(chords_by_beat, beats_per_line=4)
            
            # --- Exibir Diagramas de Acordes ---
            display_chord_diagrams(chords_by_beat)
            
            # --- Exibir Letra para Sincronização Automática Básica (Mantido) ---
            if uploaded_lyrics is not None:
                st.subheader("📝 Letra com Acordes Estimados (Pronta para Ajuste)")
                try:
                    # Lê o conteúdo do arquivo TXT
                    lyrics_content = uploaded_lyrics.getvalue().decode("utf-8")
                    
                    # Usa a função de sincronização básica
                    display_lyrics_with_chords(lyrics_content, chords_by_beat)

                except Exception as e:
                    st.error(f"Erro ao ler o arquivo de letra: {e}")

            st.markdown("---")
            
            # Visualização Gráfica
            st.subheader("📊 Visualização das Notas (Chromagram)")
            
            # CORREÇÃO: Define o limite de tempo para 60.0 segundos
            max_duration = 60.0 
            
            fig, ax = plt.subplots(figsize=(10, 5))
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
            ax.set(title='Chromagram com Batidas')
            
            # CORREÇÃO: Força o limite do eixo X para garantir a exibição total dos 60s
            ax.set_xlim([0, max_duration])
            
            # Linhas Verticais para Batidas
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            ax.vlines(beat_times, 0, chroma.shape[0], color='w', linestyle='--', alpha=0.8, label='Batidas')
            
            st.pyplot(fig) 
            
            st.markdown("---")
            st.markdown(f"**Próximo Passo:** Use a Batida (`B:xx`) ou o tempo do Chromagram para ajustar o alinhamento da letra/acordes. O Tempo é de aproximadamente **{int(tempo)} BPM**.")

        except Exception as e:
            st.error(f"Erro ao processar: {e}. (Verifique se o arquivo de áudio é válido.)")
        finally:
            # Limpeza do arquivo temporário
            if os.path.exists(tmp_path):
                 os.remove(tmp_path)
