import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.models import MultiLayerPerceptron, evaluate_predictions
from src.visualization import plot_convergence, plot_confusion_matrix
from src.preparingData import preproccessData

INK = '#0a2030'
OCEAN = '#0f2f45'
OCEAN_LIGHT = '#173f58'
OCEAN_LINE = '#285670'
ICE = '#e8f1f6'
MIST = '#93afc1'
DIM = '#5f7f93'

CYAN = '#52c6e3'
CYAN_HOVER = '#7bd6ec'
CYAN_PRESS = '#3aaecb'

GLACIER = '#eaf1f6'
SNOW = '#ffffff'
LINE = '#d3dfe8'
LINE_STRONG = '#b4c6d3'
SLATE = '#2f4a5f'
SLATE_MUTED = '#5a7385'

LOG_INFO = '#9db6c6'
LOG_TRAIN = CYAN
LOG_TEST = '#6fe0a3'
LOG_RULE = '#2b536b'

PILL_IDLE = ('#dde8ef', '#4a6577')
PILL_READY = ('#d4f2e3', '#12603b')

def _pick_font(root, candidates, fallback):
    installed = set(tkfont.families(root))
    for name in candidates:
        if name in installed:
            return name
    return fallback

class PenguinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Back-Propagation Multi-Layer Perceptron')
        self.configure(bg=GLACIER)

        height = min(950, self.winfo_screenheight() - 80)
        self.geometry(f'1380x{height}')
        self.minsize(1200, min(800, height))

        self.model = None
        self.split = None
        self.class_names = ['Adelie', 'Chinstrap', 'Gentoo']

        self._setup_fonts()
        self._setup_style()
        self._build_sidebar()
        self._build_main()
        self._show_placeholder()
        self.log('System ready. Configure MLP architecture and press Train model.', 'info')

    def _setup_fonts(self):
        ui = _pick_font(self, ['Segoe UI', 'SF Pro Text', 'Helvetica Neue', 'Inter', 'Ubuntu', 'Noto Sans'], 'Helvetica')
        mono = _pick_font(self, ['Cascadia Mono', 'Consolas', 'SF Mono', 'Menlo', 'DejaVu Sans Mono'], 'Courier')

        self.f_body = (ui, 10)
        self.f_small = (ui, 9)
        self.f_bold = (ui, 10, 'bold')
        self.f_head = (ui, 11, 'bold')
        self.f_title = (ui, 16, 'bold')
        self.f_run = (ui, 14, 'bold')
        self.f_metric = (ui, 13, 'bold')
        self.f_mono = (mono, 10)
        self.f_mono_bold = (mono, 10, 'bold')

    def _setup_style(self):
        s = ttk.Style(self)
        if 'clam' in s.theme_names():
            s.theme_use('clam')
        s.configure('.', font=self.f_body)

        field = dict(fieldbackground=OCEAN_LIGHT, foreground=ICE, bordercolor=OCEAN_LINE,
                     lightcolor=OCEAN_LINE, darkcolor=OCEAN_LINE, borderwidth=1, padding=(8, 6))
        focus = dict(bordercolor=[('focus', CYAN)], lightcolor=[('focus', CYAN)], darkcolor=[('focus', CYAN)])

        s.configure('Side.TEntry', insertcolor=ICE, **field)
        s.map('Side.TEntry', fieldbackground=[('disabled', OCEAN)], foreground=[('disabled', DIM)], **focus)

        def flat(name, bg, fg, hover, press, font, pad):
            s.configure(name, background=bg, foreground=fg, bordercolor=bg, lightcolor=bg, darkcolor=bg,
                        borderwidth=0, focusthickness=0, focuscolor=bg, padding=pad, font=font, relief='flat')
            s.map(name, background=[('pressed', press), ('active', hover)],
                  bordercolor=[('pressed', press), ('active', hover)],
                  lightcolor=[('pressed', press), ('active', hover)],
                  darkcolor=[('pressed', press), ('active', hover)])

        flat('Primary.TButton', CYAN, INK, CYAN_HOVER, CYAN_PRESS, self.f_bold, (14, 10))
        flat('Ghost.TButton', OCEAN_LIGHT, ICE, '#1f5273', OCEAN_LINE, self.f_body, (12, 8))
        flat('Link.TButton', INK, MIST, INK, INK, self.f_small, (8, 2))
        s.map('Link.TButton', foreground=[('active', ICE)])

        s.configure('Seg.Toolbutton', background=OCEAN_LIGHT, foreground=MIST, bordercolor=OCEAN_LIGHT,
                    lightcolor=OCEAN_LIGHT, darkcolor=OCEAN_LIGHT, borderwidth=0, focusthickness=0,
                    focuscolor=OCEAN_LIGHT, padding=(10, 8), anchor='center', font=self.f_bold, relief='flat')
        s.map('Seg.Toolbutton', background=[('selected', CYAN), ('active', '#1f5273')],
              foreground=[('selected', INK), ('active', ICE)],
              bordercolor=[('selected', CYAN)], lightcolor=[('selected', CYAN)], darkcolor=[('selected', CYAN)],
              relief=[('selected', 'flat'), ('pressed', 'flat')])

        s.configure('Side.TCheckbutton', background=OCEAN, foreground=ICE, focuscolor=OCEAN,
                    indicatorbackground=OCEAN_LIGHT, indicatorforeground=INK,
                    upperbordercolor=OCEAN_LINE, lowerbordercolor=OCEAN_LINE, padding=(2, 2))
        s.map('Side.TCheckbutton', background=[('active', OCEAN)],
              indicatorbackground=[('selected', CYAN), ('active', OCEAN_LINE)],
              indicatorforeground=[('selected', INK)])

        s.configure('Console.Vertical.TScrollbar', background=OCEAN_LINE, troughcolor=INK, bordercolor=INK,
                    lightcolor=OCEAN_LINE, darkcolor=OCEAN_LINE, arrowcolor=MIST, relief='flat', gripcount=0)
        s.map('Console.Vertical.TScrollbar', background=[('active', '#3a6c87')])

    def _section(self, parent, title):
        tk.Frame(parent, bg=OCEAN_LINE, height=1).pack(fill='x', pady=(12, 8))
        tk.Label(parent, text=title, bg=OCEAN, fg=ICE, font=self.f_head, anchor='w').pack(fill='x', pady=(0, 6))
        grid = tk.Frame(parent, bg=OCEAN)
        grid.pack(fill='x')
        grid.columnconfigure((0, 1), weight=1, uniform='side')
        return grid

    def _labeled(self, grid, text, row, col, span=1):
        padx = (0, 0) if span == 2 else ((0, 6) if col == 0 else (6, 0))
        cell = tk.Frame(grid, bg=OCEAN)
        cell.grid(row=row, column=col, columnspan=span, sticky='ew', padx=padx, pady=(0, 8))
        caption = tk.Label(cell, text=text, bg=OCEAN, fg=MIST, font=self.f_small, anchor='w')
        caption.pack(fill='x', pady=(0, 3))
        return cell, caption

    def _entry(self, parent, default=''):
        ent = ttk.Entry(parent, style='Side.TEntry', width=10)
        if default:
            ent.insert(0, default)
        ent.pack(fill='x')
        return ent

    def _build_sidebar(self):
        side = tk.Frame(self, bg=OCEAN, width=336)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)
        body = tk.Frame(side, bg=OCEAN)
        body.pack(fill='both', expand=True, padx=22, pady=12)
        tk.Label(body, text='Back-Propagation', bg=OCEAN, fg=ICE, font=self.f_title, anchor='w').pack(fill='x', pady=(0, 5))

        grid = self._section(body, 'Architecture Config')
        cell, _ = self._labeled(grid, 'Number of hidden layers', 0, 0, span=2)
        self.ent_num_layers = self._entry(cell, '1')

        cell, _ = self._labeled(grid, 'Neurons (comma-sep)', 1, 0, span=2)
        self.ent_nodes = self._entry(cell, '5')
        grid = self._section(body, 'Hyperparameters')
        cell, _ = self._labeled(grid, 'Learning rate (eta)', 0, 0)
        self.ent_eta = self._entry(cell, '0.01')

        cell, _ = self._labeled(grid, 'Epochs (m)', 0, 1)
        self.ent_epochs = self._entry(cell, '1000')

        self.var_act = tk.StringVar(value='Sigmoid')
        seg = tk.Frame(grid, bg=OCEAN)
        seg.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 10))
        seg.columnconfigure((0, 1), weight=1, uniform='seg')

        ttk.Radiobutton(seg, text='Sigmoid', value='Sigmoid', variable=self.var_act, style='Seg.Toolbutton').grid(row=0, column=0, sticky='ew')
        ttk.Radiobutton(seg, text='Tanh', value='Tanh', variable=self.var_act, style='Seg.Toolbutton').grid(row=0, column=1, sticky='ew')

        self.var_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text='Include bias', variable=self.var_bias, style='Side.TCheckbutton').grid(row=2, column=0, sticky='sw', pady=(0, 12))

        grid = self._section(body, 'Execution')
        ttk.Button(grid, text='Train MLP Model', style='Primary.TButton', command=self.on_train).grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Button(grid, text='Test & Evaluate', style='Ghost.TButton', command=self.on_test).grid(row=1, column=0, sticky='ew', padx=(0, 4))
        ttk.Button(grid, text='Save plots', style='Ghost.TButton', command=self.on_save).grid(row=1, column=1, sticky='ew', padx=(4, 0))

    @staticmethod
    def _panel(parent):
        outer = tk.Frame(parent, bg=LINE)
        inner = tk.Frame(outer, bg=SNOW)
        inner.pack(fill='both', expand=True, padx=1, pady=1)
        return outer, inner

    def _build_main(self):
        main = tk.Frame(self, bg=GLACIER)
        main.pack(side='left', fill='both', expand=True, padx=24, pady=(18, 20))

        header = tk.Frame(main, bg=GLACIER)
        header.pack(fill='x')
        titles = tk.Frame(header, bg=GLACIER)
        titles.pack(side='left', fill='x', expand=True)

        self.var_run_title = tk.StringVar(value='No MLP trained yet')
        self.var_run_sub = tk.StringVar(value='Your convergence and confusion matrix will appear here.')
        tk.Label(titles, textvariable=self.var_run_title, bg=GLACIER, fg=INK, font=self.f_run, anchor='w').pack(fill='x')
        tk.Label(titles, textvariable=self.var_run_sub, bg=GLACIER, fg=SLATE_MUTED, font=self.f_body, anchor='w').pack(fill='x')

        self.pill = tk.Label(header, text='Not trained', bg=PILL_IDLE[0], fg=PILL_IDLE[1], font=self.f_bold, padx=14, pady=5)
        self.pill.pack(side='right', anchor='n')

        outer, strip = self._panel(main)
        outer.pack(fill='x', pady=(14, 12))

        self.var_m_arch = tk.StringVar(value='-')
        self.var_m_epochs = tk.StringVar(value='-')
        self.var_m_tr_acc = tk.StringVar(value='-')
        self.var_m_te_acc = tk.StringVar(value='-')

        metrics = (('Hidden Architecture', self.var_m_arch), ('Epochs run', self.var_m_epochs),
                   ('Final Train Acc.', self.var_m_tr_acc), ('Test Accuracy', self.var_m_te_acc))

        for i, (caption, var) in enumerate(metrics):
            if i:
                tk.Frame(strip, bg=LINE, width=1).grid(row=0, column=2 * i - 1, sticky='ns', pady=10)
            cell = tk.Frame(strip, bg=SNOW)
            cell.grid(row=0, column=2 * i, sticky='ew', padx=18, pady=10)
            strip.columnconfigure(2 * i, weight=1, uniform='metric')
            tk.Label(cell, text=caption, bg=SNOW, fg=SLATE_MUTED, font=self.f_small, anchor='w').pack(fill='x')
            tk.Label(cell, textvariable=var, bg=SNOW, fg=INK, font=self.f_metric, anchor='w').pack(fill='x')

        console = tk.Frame(main, bg=INK)
        console.pack(side='bottom', fill='x', pady=(12, 0))
        bar = tk.Frame(console, bg=INK)
        bar.pack(fill='x', padx=14, pady=(8, 0))
        tk.Label(bar, text='Activity log', bg=INK, fg=MIST, font=self.f_bold).pack(side='left')
        ttk.Button(bar, text='Clear', style='Link.TButton', command=self._clear_log).pack(side='right')

        text_row = tk.Frame(console, bg=INK)
        text_row.pack(fill='x', padx=(14, 4), pady=(2, 10))
        self.log_box = tk.Text(text_row, height=5, wrap='none', state='disabled', font=self.f_mono, bg=INK, fg=ICE,
                               relief='flat', borderwidth=0, highlightthickness=0, padx=0, pady=4,
                               insertbackground=ICE, selectbackground=OCEAN_LINE, spacing1=1, spacing3=1)
        sb = ttk.Scrollbar(text_row, orient='vertical', command=self.log_box.yview, style='Console.Vertical.TScrollbar')
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log_box.pack(side='left', fill='both', expand=True)

        self.log_box.tag_config('info', foreground=LOG_INFO)
        self.log_box.tag_config('rule', foreground=LOG_RULE)
        self.log_box.tag_config('train', foreground=LOG_TRAIN, font=self.f_mono_bold)
        self.log_box.tag_config('test', foreground=LOG_TEST, font=self.f_mono_bold)
        self.log_box.tag_config('matrix', foreground=ICE, font=self.f_mono)

        outer, plot_panel = self._panel(main)
        outer.pack(side='top', fill='both', expand=True)

        try:
            self.fig = Figure(figsize=(12, 10), dpi=100, layout='constrained')
        except TypeError:
            self.fig = Figure(figsize=(12, 10), dpi=100, constrained_layout=True)

        self.fig.patch.set_facecolor(SNOW)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        self.ax_convergence = self.fig.add_subplot(gs[0])
        self.ax_confusion = self.fig.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas.get_tk_widget().configure(bg=SNOW, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

    def _show_placeholder(self):
        messages = ((self.ax_convergence, 'Convergence curve will display here.'),
                    (self.ax_confusion, 'Confusion Matrix will display here after Testing.'))
        for ax, msg in messages:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
                    color=SLATE_MUTED, fontsize=11, linespacing=1.5)
        self.canvas.draw()

    def _set_status(self, text, ready):
        bg, fg = PILL_READY if ready else PILL_IDLE
        self.pill.configure(text=text, bg=bg, fg=fg)

    def log(self, msg, tag='info'):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', msg + '\n', tag)
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _clear_log(self):
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')

    def _read_params(self):
        try:
            eta = float(self.ent_eta.get())
            epochs = int(self.ent_epochs.get())
            num_layers = int(self.ent_num_layers.get())
            nodes_str = self.ent_nodes.get().strip()

            if not nodes_str:
                raise ValueError("Nodes cannot be empty")

            nodes_list = [int(x.strip()) for x in nodes_str.split(',')]

            if len(nodes_list) != num_layers:
                messagebox.showerror('Input error', f"You specified {num_layers} layers but provided {len(nodes_list)} node counts.")
                return None

            if eta <= 0 or epochs <= 0:
                raise ValueError("Values must be > 0")

        except ValueError:
            messagebox.showerror('Input error', 'Please ensure all inputs are valid positive numbers.')
            return None

        return dict(eta=eta, epochs=epochs, num_layers=num_layers, hidden_layers=nodes_list,
                    use_bias=self.var_bias.get(), activation=self.var_act.get())

    def on_train(self):
        p = self._read_params()
        if p is None:
            return

        X_tr, y_tr, X_te, y_te = preproccessData('penguins.csv')

        self.model = MultiLayerPerceptron(
            hidden_layers=p['hidden_layers'],
            activation=p['activation'],
            learning_rate=p['eta'],
            epochs=p['epochs'],
            use_bias=p['use_bias']
        )

        self.model.fit(X_tr, y_tr)
        self.split = (X_tr, y_tr, X_te, y_te)
        self.ax_convergence.clear()
        self.ax_convergence.set_axis_on()

        plot_convergence(self.model, ax=self.ax_convergence)
        self.canvas.draw()

        final_acc = self.model.accuracy_history[-1] * 100

        bias_text = 'with bias' if p['use_bias'] else 'without bias'
        arch_str = f"5 -> {', '.join(map(str, p['hidden_layers']))} -> 3"

        self.var_run_title.set(f"MLP with {p['activation']} Activation")
        self.var_run_sub.set(f"Architecture: [{arch_str}], {bias_text}, learning rate {p['eta']}")

        self.var_m_arch.set(arch_str)
        self.var_m_epochs.set(f"{p['epochs']}")
        self.var_m_tr_acc.set(f"{final_acc:.1f}%")
        self.var_m_te_acc.set('Not tested yet')

        self._set_status('Model trained', True)

        self.log('─' * 75, 'rule')
        self.log(f"[Train] Multi-Layer Perceptron ({p['activation']})", 'train')
        self.log(f"Architecture: {arch_str} | eta={p['eta']} | bias={p['use_bias']} | epochs={p['epochs']}", 'info')
        self.log(f"Final training accuracy = {final_acc:.2f}%", 'info')

    def on_test(self):
        if self.model is None:
            messagebox.showwarning('Not trained', 'Train a model first.')
            return

        _, _, X_te, y_te = self.split
        y_pred = self.model.predict_classes(X_te)

        cm, acc = evaluate_predictions(y_te, y_pred)

        self.var_m_te_acc.set(f'{acc * 100:.1f}%')
        self.ax_confusion.clear()
        self.ax_confusion.set_axis_on()
        plot_confusion_matrix(cm, self.class_names, ax=self.ax_confusion)
        self.canvas.draw()

        self.log('─' * 75, 'rule')
        self.log(f"[Test] Evaluation on {len(y_te)} test samples", 'test')
        self.log(f"Overall test accuracy = {acc * 100:.2f}%\n", 'info')

    def on_save(self):
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile='mlp_results.png',
                                            filetypes=[('PNG image', '*.png')])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            self.log('─' * 75, 'rule')
            self.log(f"[Save] Figure saved to: {path}", 'info')

if __name__ == '__main__':
    PenguinGUI().mainloop()