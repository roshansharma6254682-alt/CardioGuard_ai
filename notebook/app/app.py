import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
def analyze_ecg(sample_index):
    sample_index = int(sample_index)
    ecg = X_test[sample_index]
    result = predict_heartbeat(ecg)
    actual = "Abnormal" if y_test[sample_index] == 1 else "Normal"
    time = np.arange(len(ecg)) / 360
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, ecg, linewidth=1.8)
    ax.set_title("ECG Heartbeat Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return (
        result["Prediction"],
        f'{result["Abnormal Probability"] * 100:.2f}%',
        actual,
        fig)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ❤️ CardioGuard_ai
    ### AI-Based ECG Heartbeat Classification
    Analyze ECG heartbeat segments and view the model's prediction.
    """)
    with gr.Row():
        sample_input = gr.Slider(
            minimum=0,
            maximum=len(X_test) - 1,
            value=0,
            step=1,
            label="Select ECG Sample")
    analyze_btn = gr.Button("🔍 Analyze ECG", variant="primary")

    with gr.Row():
        prediction = gr.Textbox(label="Predicted Class")
        probability = gr.Textbox(label="Abnormal Probability")
        actual = gr.Textbox(label="Dataset Label")
    graph = gr.Plot(label="ECG Waveform")
    analyze_btn.click(
        fn=analyze_ecg,
        inputs=sample_input,
        outputs=[prediction, probability, actual, graph]
    )
    gr.Markdown("""
    ---
    **Disclaimer:** This is an educational research prototype.
    It is not validated for clinical diagnosis or medical decisions.
    """)
demo.launch(share=True)