import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class InteractiveBlochSphere:
    r"""Interactive Bloch Sphere class

    Created to help visualize the transformations of quantum states in the
    Bloch sphere representation. Allows the user to change the values of
    :math:`\theta` and :math:`\phi` for:

    .. math::
       \left|\psi\right> = \cos\frac{\theta}{2}\left|0\right> + e^{i\phi}
       \sin \frac{\theta}{2}\left|1\right>

    """

    def run(self):
        """Run the interactive Bloch Sphere in a Tkinter window"""
        # Root window
        matplotlib.use('TkAgg')
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.geometry('680x540')
        self.root.resizable(True, True)
        self.root.title('Bloch Sphere')

        # Variables for the angles
        self.theta = tk.DoubleVar()
        self.phi = tk.DoubleVar()

        # Bloch Sphere figure canvas
        base_state = Statevector(np.array([1, 0]))
        base_figure = base_state.draw('bloch')
        self.figure_canvas = FigureCanvasTkAgg(base_figure, self.root)
        self.figure_canvas.get_tk_widget().grid(column=1, row=1)

        # Configuration of Theta slider
        self.theta_label = ttk.Label(self.root, text=u'\u03B8=0.00rad', font=("Arial", 15),
                                anchor='center')
        self.theta_label.grid(column=0, row=0, sticky='we')
        slider_theta = ttk.Scale(self.root, from_=0.0, to=np.pi,
                                 orient='vertical',
                                 command=self._plot_update,
                                 variable=self.theta)
        slider_theta.grid(column=0, row=1, sticky='ns')

        # Configuration of Phi slider
        self.phi_label = ttk.Label(self.root, text=u'\u03D5=0.00rad', font=("Arial", 15),
                              anchor='e')
        self.phi_label.grid(column=2, row=2, sticky='e')
        slider_phi = ttk.Scale(self.root, from_=0.0, to=2 * np.pi,
                               orient='horizontal',
                               command=self._plot_update,
                               variable=self.phi)
        slider_phi.grid(column=1, row=2, sticky='nesw')

        # Configuration of state text label
        self.state = ttk.Label(self.root, text=u'1.00|0\u29FD+'
                                               u'(0.00+0.00j)|1\u29FD',
                               font=("Arial", 15), anchor='center')
        self.state.grid(column=1, row=0, sticky='we')

        self.root.mainloop()

        return

    def _plot_update(self, event):
        # Update labels with information about angles
        self.theta_label['text'] = u'\u03B8={:.2f}rad'.format(self.theta.get())
        self.phi_label['text'] = u'\u03D5={:.2f}rad'.format(self.phi.get())

        # Calculate amplitude vector
        amp_0 = np.cos(self.theta.get() / 2)
        amp_1 = np.sin(self.theta.get() / 2) * np.exp(1j * self.phi.get())
        self.state['text'] = u'{:.2f}|0\u276D+({:.2f})|1\u276D'.format(amp_0,
                                                                       amp_1)

        # Generate Bloch Sphere figure
        state = Statevector(np.array([amp_0, amp_1]))
        figure = state.draw('bloch')
        plt.close(self.figure_canvas.figure)
        self.figure_canvas.get_tk_widget().destroy()
        self.figure_canvas = FigureCanvasTkAgg(figure, self.root)
        self.figure_canvas.get_tk_widget().grid(column=1, row=1)

    def _quit(self):
        self.root.quit()
        self.root.destroy()
