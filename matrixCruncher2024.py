import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numba import threading_layer, config, set_num_threads, njit
import threading
import os
import time


from scipy.integrate import quad

# Configure Numba threading layer
config.THREADING_LAYER = 'omp'  # Or 'tbb' for Intel Threading Building Blocks
set_num_threads(os.cpu_count())  # Use all available CPU cores

class MatrixApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Matrix2024")
        self.geometry("500x450")

        # Main Menu
        self.create_main_menu()

    def show_threading_layer(self):
        threading_info = f"Threading layer chosen: {threading_layer()}"
        tk.messagebox.showinfo("Threading Layer", threading_info)

    def create_main_menu(self):
        tk.Label(self, text="Choose Operation", font=("Arial", 14)).pack(pady=10)
        options = ["Matrix Multiplication", "LU Decomposition", "Linear Regression", "Lagrange Interpolation Polynom", "Least Square Solution", "Probability Density Function", "Multi-Core Fp", "Multi-Core Int"]
        for option in options:
            button = tk.Button(self, text=option, width=25, command=lambda opt=option: self.open_option_window(opt))
            button.pack(pady=5)

        # Button to show threading layer
        threading_button = tk.Button(self, text="Show Threading Layer", width=25, command=self.show_threading_layer)
        threading_button.pack(pady=10)

    def open_option_window(self, option):
        if option == "Matrix Multiplication":
            window = MatrixMultiplier(self)
            self.position_window(window, "top_left")
        elif option == "LU Decomposition":
            window = LUDecomposition(self)
            self.position_window(window, "top_right")
        elif option == "Linear Regression":
            window = LinearRegression(self)
            self.position_window(window, "bottom_left")
        elif option == "Least Square Solution":
            window = LeastSquareSolution(self)
            self.position_window(window, "bottom_right")
        elif option == "Probability Density Function":
            window = Probability_Density_Function(self)
            self.position_window(window, "center")
        elif option == "Lagrange Interpolation Polynom":
            window = LagrangeInterpolation(self)
            self.position_window(window, "center")
        elif option == "Multi-Core Fp":
            window = CpuBenchmarkFp(self)
        elif option == "Multi-Core Int":
            window = CpuBenchmarkInteger(self)

    def position_window(self, window, position):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        if position == "top_left":
            x = 0
            y = 0
        elif position == "top_right":
            x = screen_width - width
            y = 0
        elif position == "bottom_left":
            x = 0
            y = screen_height - height - 50  # Adjust for taskbar
        elif position == "center":
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

        window.geometry(f"+{x}+{y}")

class MatrixMultiplier(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Matrix Multiplication")
        self.geometry("600x400")

        # Frames to organize the layout
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, pady=10)

        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.pack(side=tk.TOP)

        self.create_input_fields()

    def create_input_fields(self):
        # Input for Matrix A dimensions
        tk.Label(self.top_frame, text="Matrix A Rows:").grid(row=0, column=0)
        self.rows_a_entry = tk.Entry(self.top_frame, width=5)
        self.rows_a_entry.grid(row=0, column=1)

        tk.Label(self.top_frame, text="Columns:").grid(row=0, column=2)
        self.cols_a_entry = tk.Entry(self.top_frame, width=5)
        self.cols_a_entry.grid(row=0, column=3)

        # Input for Matrix B dimensions
        tk.Label(self.top_frame, text="Matrix B Rows:").grid(row=1, column=0)
        self.rows_b_entry = tk.Entry(self.top_frame, width=5)
        self.rows_b_entry.grid(row=1, column=1)

        tk.Label(self.top_frame, text="Columns:").grid(row=1, column=2)
        self.cols_b_entry = tk.Entry(self.top_frame, width=5)
        self.cols_b_entry.grid(row=1, column=3)

        # Button to create matrix input fields
        self.create_matrices_button = tk.Button(
            self.top_frame, text="Create Matrices", command=self.create_matrix_inputs)
        self.create_matrices_button.grid(row=2, column=1, columnspan=2, pady=5)

    def create_matrix_inputs(self):
        # Clear previous matrices if any
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        try:
            self.rows_a = int(self.rows_a_entry.get())
            self.cols_a = int(self.cols_a_entry.get())
            self.rows_b = int(self.rows_b_entry.get())
            self.cols_b = int(self.cols_b_entry.get())

            if self.cols_a != self.rows_b:
                messagebox.showerror("Error", "Number of columns in Matrix A must equal number of rows in Matrix B.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer dimensions.")
            return

        self.matrix_a_entries = []
        self.matrix_b_entries = []

        # Matrix A
        tk.Label(self.matrix_frame, text="Matrix A").grid(row=0, column=0, columnspan=self.cols_a)
        for i in range(self.rows_a):
            row_entries = []
            for j in range(self.cols_a):
                entry = tk.Entry(self.matrix_frame, width=7)
                entry.grid(row=i+1, column=j)
                row_entries.append(entry)
            self.matrix_a_entries.append(row_entries)

        # Spacer
        tk.Label(self.matrix_frame, text="").grid(row=0, column=self.cols_a)

        # Matrix B
        tk.Label(self.matrix_frame, text="Matrix B").grid(row=0, column=self.cols_a+1, columnspan=self.cols_b)
        for i in range(self.rows_b):
            row_entries = []
            for j in range(self.cols_b):
                entry = tk.Entry(self.matrix_frame, width=7)
                entry.grid(row=i+1, column=self.cols_a+1+j)
                row_entries.append(entry)
            self.matrix_b_entries.append(row_entries)

        # Compute button
        self.compute_button = tk.Button(
            self.matrix_frame, text="Compute Product", command=self.compute_product)
        self.compute_button.grid(row=max(self.rows_a, self.rows_b)+1, column=0, columnspan=self.cols_a+self.cols_b+1, pady=10)

    def compute_product(self):
        # Get matrices from entries
        try:
            A = np.array([[float(entry.get()) for entry in row] for row in self.matrix_a_entries])
            B = np.array([[float(entry.get()) for entry in row] for row in self.matrix_b_entries])
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in the matrices.")
            return

        # Multiply matrices using Numba-accelerated function
        try:
            result = matrix_multiply_numba(A, B)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during multiplication: {e}")
            return

        # Display result in a new window
        self.display_result_window(result)

    def display_result_window(self, result):
        result_window = tk.Toplevel(self)
        result_window.title("Result Matrix")

        # Adjust window size based on matrix dimensions
        window_width = min(1000, 100 * result.shape[1])
        window_height = min(800, 30 * result.shape[0])
        result_window.geometry(f"{window_width}x{window_height}")

        # Create a canvas and scrollbar to handle large matrices
        canvas = tk.Canvas(result_window)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(result_window, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        result_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=result_frame, anchor='nw')

        # Update scrollregion after widgets are added
        result_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        # Display the result matrix
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                label = tk.Label(result_frame, text=str(result[i, j]), width=15, borderwidth=1, relief="solid", anchor='e')
                label.grid(row=i, column=j, padx=1, pady=1)

class LUDecomposition(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("LU Decomposition")
        self.geometry("600x400")

        # Frame for input
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(pady=10)

        # Frame for matrix entries
        self.matrix_frame = tk.Frame(self)
        self.matrix_frame.pack()

        self.create_input_fields()

    def create_input_fields(self):
        # Input for Matrix dimensions
        tk.Label(self.input_frame, text="Matrix Size (n x n):").grid(row=0, column=0)
        self.size_entry = tk.Entry(self.input_frame, width=5)
        self.size_entry.grid(row=0, column=1)

        # Button to create matrix input fields
        self.create_matrix_button = tk.Button(
            self.input_frame, text="Create Matrix", command=self.create_matrix_input)
        self.create_matrix_button.grid(row=1, column=0, columnspan=2, pady=5)

    def create_matrix_input(self):
        # Clear previous matrix if any
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        try:
            self.size = int(self.size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for size.")
            return

        self.matrix_entries = []

        # Matrix A
        tk.Label(self.matrix_frame, text="Matrix A").grid(row=0, column=0, columnspan=self.size)
        for i in range(self.size):
            row_entries = []
            for j in range(self.size):
                entry = tk.Entry(self.matrix_frame, width=7)
                entry.grid(row=i+1, column=j)
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)

        # Compute button
        self.compute_button = tk.Button(
            self.matrix_frame, text="Compute LU Decomposition", command=self.compute_lu)
        self.compute_button.grid(row=self.size+1, column=0, columnspan=self.size, pady=10)

    def compute_lu(self):
        # Get matrix from entries
        try:
            A = np.array([[float(entry.get()) for entry in row] for row in self.matrix_entries])
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in the matrix.")
            return

        # Compute LU Decomposition
        try:
            L, U = lu_decomposition(A)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during LU Decomposition: {e}")
            return

        # Display result in a new window
        self.display_lu_result(L, U)

    def display_lu_result(self, L, U):
        result_window = tk.Toplevel(self)
        result_window.title("LU Decomposition Result")

        # L matrix
        tk.Label(result_window, text="L Matrix").grid(row=0, column=0, columnspan=self.size)
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                label = tk.Label(result_window, text=f"{L[i, j]:.4f}", width=10, borderwidth=1, relief="solid", anchor='e')
                label.grid(row=i+1, column=j)

        # U matrix
        tk.Label(result_window, text="U Matrix").grid(row=0, column=self.size+1, columnspan=self.size)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                label = tk.Label(result_window, text=f"{U[i, j]:.4f}", width=10, borderwidth=1, relief="solid", anchor='e')
                label.grid(row=i+1, column=self.size+1+j)

class LinearRegression(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Linear Regression")
        self.geometry("600x700")

        # Frame for inputs
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(pady=10)

        self.create_input_fields()

    def create_input_fields(self):
        tk.Label(self.input_frame, text="Enter data points (x,y per line) or generate data:").pack()

        # Frame for data generator parameters
        generator_frame = tk.Frame(self.input_frame)
        generator_frame.pack(pady=5)

        # Number of points
        tk.Label(generator_frame, text="Number of Points:").grid(row=0, column=0, sticky='e')
        self.num_points_entry = tk.Entry(generator_frame, width=5)
        self.num_points_entry.insert(0, "50")
        self.num_points_entry.grid(row=0, column=1)

        # Slope
        tk.Label(generator_frame, text="True Slope:").grid(row=1, column=0, sticky='e')
        self.true_slope_entry = tk.Entry(generator_frame, width=5)
        self.true_slope_entry.insert(0, "1.0")
        self.true_slope_entry.grid(row=1, column=1)

        # Intercept
        tk.Label(generator_frame, text="True Intercept:").grid(row=2, column=0, sticky='e')
        self.true_intercept_entry = tk.Entry(generator_frame, width=5)
        self.true_intercept_entry.insert(0, "0.0")
        self.true_intercept_entry.grid(row=2, column=1)

        # Noise Level
        tk.Label(generator_frame, text="Noise Level (Std Dev):").grid(row=3, column=0, sticky='e')
        self.noise_level_entry = tk.Entry(generator_frame, width=5)
        self.noise_level_entry.insert(0, "1.0")
        self.noise_level_entry.grid(row=3, column=1)

        # Generate Data Button
        self.generate_button = tk.Button(
            generator_frame, text="Generate Data", command=self.generate_data)
        self.generate_button.grid(row=4, column=0, columnspan=2, pady=5)

        # Create a Text widget with scrollbars
        text_frame = tk.Frame(self.input_frame)
        text_frame.pack(pady=5)

        self.data_text = tk.Text(text_frame, width=50, height=15)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.data_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text.configure(yscrollcommand=scrollbar.set)

        # Instruction Label
        instructions = (
            "Enter one data point per line, with x and y values separated by a comma.\n"
            "Example:\n"
            "1, 2\n"
            "2, 3\n"
            "3, 5\n"
            "Alternatively, use the data generator above."
        )
        tk.Label(self.input_frame, text=instructions, justify=tk.LEFT).pack()

        # Compute button
        self.compute_button = tk.Button(
            self.input_frame, text="Compute Regression", command=self.compute_regression)
        self.compute_button.pack(pady=10)

    def generate_data(self):
        try:
            num_points = int(self.num_points_entry.get())
            true_slope = float(self.true_slope_entry.get())
            true_intercept = float(self.true_intercept_entry.get())
            noise_level = float(self.noise_level_entry.get())

            if num_points < 2 or num_points > 100_000_000:
                messagebox.showerror("Error", "Number of points must be between 2 and 1000.")
                return

            # Generate x values uniformly between -10 and 10
            x_values = np.linspace(-10, 10, num_points)

            # Generate y values with noise
            noise = np.random.normal(0, noise_level, num_points)
            y_values = true_slope * x_values + true_intercept + noise

            # Clear the text widget
            self.data_text.delete("1.0", tk.END)

            # Insert generated data into the text widget
            for x, y in zip(x_values, y_values):
                self.data_text.insert(tk.END, f"{x:.4f}, {y:.4f}\n")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for all parameters.")

    def compute_regression(self):
        data = self.data_text.get("1.0", tk.END).strip()
        lines = data.splitlines()
        if not lines:
            messagebox.showerror("Error", "Please enter or generate data points.")
            return

        x_values = []
        y_values = []

        for line_num, line in enumerate(lines, start=1):
            if line.strip() == '':
                continue  # Skip empty lines
            try:
                x_str, y_str = line.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                x_values.append(x)
                y_values.append(y)
            except ValueError:
                messagebox.showerror("Error", f"Invalid format on line {line_num}. Expected 'x, y'.")
                return

        if len(x_values) != len(y_values):
            messagebox.showerror("Error", "Number of x and y values must be the same.")
            return
        if len(x_values) < 2:
            messagebox.showerror("Error", "At least two data points are required.")
            return

        # Convert to NumPy arrays with float64 data type for Numba compatibility
        x_array = np.array(x_values, dtype=np.float64)
        y_array = np.array(y_values, dtype=np.float64)

        # Compute regression coefficients
        try:
            slope, intercept = linear_regression(x_array, y_array)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during regression: {e}")
            return

        # Display results and plot
        self.display_regression_result(x_values, y_values, slope, intercept)


    def display_regression_result(self, x_values, y_values, slope, intercept):
        result_window = tk.Toplevel(self)
        result_window.title("Linear Regression Result")
        result_window.geometry("600x600")

        # Display equation
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
        tk.Label(result_window, text="Regression Equation:", font=("Arial", 12)).pack(pady=10)
        tk.Label(result_window, text=equation, font=("Arial", 12)).pack()

        # Plot the data and regression line
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x_values, y_values, color='blue', label='Data Points')
        x_line = np.linspace(min(x_values), max(x_values), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='red', label='Regression Line')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Linear Regression')

        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

class LagrangeInterpolation(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Lagrange Interpolation Polynom")
        self.geometry("600x400")

        # Frame for input
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(pady=10)

        # Frame for matrix entries
        self.entries= {}

        self.create_input_fields()

    def create_input_fields(self):
        # Input for lagrange x0 x1 f0 f1 etc values
        num_points = 8

        for i in range(num_points):
            # Create label and entry for xi
            x_label = tk.Label(self.input_frame, text=f"x{i}", font=("Arial", 12))
            x_label.grid(row=i, column=0, padx=5, pady=5)
            x_entry = tk.Entry(self.input_frame, font=("Arial", 12), width=10)
            x_entry.grid(row=i, column=1, padx=5, pady=5)

            # Store the entry in the dictionary
            self.entries[f"x{i}"] = x_entry

            # Create label and entry for fi
            f_label = tk.Label(self.input_frame, text=f"f{i}", font=("Arial", 12))
            f_label.grid(row=i, column=2, padx=5, pady=5)
            f_entry = tk.Entry(self.input_frame, font=("Arial", 12), width=10)
            f_entry.grid(row=i, column=3, padx=5, pady=5)

            # Store the entry in the dictionary
            self.entries[f"f{i}"] = f_entry

        # Compute button
        self.compute_button = tk.Button(
        self, text="Compute Lagrange Polynomial", command=self.compute_Lagrange)
        self.compute_button.pack(pady=10)

    def compute_Lagrange(self):

        xi = []
        fi = []

        for i in range(len(self.entries)// 2):
            x_entry = self.entries.get(f"x{i}")
            f_entry = self.entries.get(f"f{i}")
            x_val = x_entry.get().strip()
            y_val = f_entry.get().strip()

            if x_val == y_val:
                continue

            try:
                x = np.float16(x_val)
                y = np.float16(y_val)
                xi.append(x)
                fi.append(y)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid input at x{i} or  f{i}. Please enter numerical values.")

        if len(xi)< 2:
            messagebox.showerror("Error", "At least two datapoints are required.")
            return

        if len(set(xi)) != len(xi):
            messagebox.showerror("Input Error", "x values must be unique.")
            return
        #convert to np array for numba comp
        xi = np.array(xi, dtype=np.float64)
        fi = np.array(fi, dtype=np.float64)

        #Define the X values where we want to eval
        x_min, x_max = np.min(xi), np.max(xi)
        x_range = x_max -x_min
        x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 500)

        y_plot = compute_lagrange_polynomial_numba(xi, fi, x_plot)

        plot_window = tk.Toplevel(self)
        plot_window.title("Lagrange Interpolation Polynomial")

        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x_plot, y_plot, label='Interpolation Polynomial')
        ax.scatter(xi, fi, color='red', label='Data Points')
        ax.set_xlabel('x')
        ax.set_ylabel('P(x)')
        ax.set_title('Lagrange Interpolation Polynomial')
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

class LeastSquareSolution(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Least Square Solution")
        self.geometry("600x600")

        # Frame for inputs
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(pady=10)

        self.create_input_fields()

    def create_input_fields(self):
        tk.Label(self.input_frame, text="Enter equations manually or load from a file:").pack()

        # Create a Text widget with scrollbar
        text_frame = tk.Frame(self.input_frame)
        text_frame.pack(pady=5)

        self.data_text = tk.Text(text_frame, width=50, height=15)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.data_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text.configure(yscrollcommand=scrollbar.set)

        # Instruction Label
        instructions = (
            "Enter one equation per line, with coefficients and constant separated by spaces or commas.\n"
            "Example:\n"
            "1 2 3    # Represents 1*x1 + 2*x2 = 3\n"
            "2 3 4    # Represents 2*x1 + 3*x2 = 4\n"
            "3 5 -5   # Represents 3*x1 + 5*x2 = -5\n"
            "Alternatively, you can load a file containing the equations."
        )
        tk.Label(self.input_frame, text=instructions, justify=tk.LEFT).pack()

        # Button to load from file
        self.load_file_button = tk.Button(
            self.input_frame, text="Load Equations from File", command=self.load_equations_from_file)
        self.load_file_button.pack(pady=5)

        # Label to display selected file
        self.selected_file_label = tk.Label(self.input_frame, text="")
        self.selected_file_label.pack()

        # Compute Button
        self.compute_button = tk.Button(
            self.input_frame, text="Compute Least Square Solution", command=self.compute_least_square_solution)
        self.compute_button.pack(pady=10)

        # Variable to store file path
        self.file_path = None

    def load_equations_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Equations File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.selected_file_label.config(text=f"Selected File: {file_path}")
            self.data_text.delete("1.0", tk.END)  # Clear the text widget
        else:
            self.file_path = None
            self.selected_file_label.config(text="")

    def compute_least_square_solution(self):
        if self.file_path:
            # Process the file without loading entire content into memory
            try:
                # Process the large file in a separate thread to keep the UI responsive
                threading.Thread(target=self.process_large_file).start()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while processing the file: {e}")
                return
        else:
            # Process data from the text widget
            data = self.data_text.get("1.0", tk.END).strip()
            lines = data.splitlines()
            if not lines:
                messagebox.showerror("Error", "Please enter at least one equation or load a file.")
                return

            A_rows = []
            b_values = []
            for line_num, line in enumerate(lines, start=1):
                if line.strip() == '':
                    continue  # Skip empty lines
                try:
                    # Split the line into numbers
                    numbers = line.replace(',', ' ').split()
                    numbers = [float(num) for num in numbers]
                    if len(numbers) < 2:
                        messagebox.showerror("Error", f"Invalid input on line {line_num}. Each line must contain at least one coefficient and one constant.")
                        return
                    # The last number is the constant term b_i
                    b_i = numbers[-1]
                    a_i = numbers[:-1]
                    A_rows.append(a_i)
                    b_values.append(b_i)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid number format on line {line_num}.")
                    return

            A = np.array(A_rows, dtype=np.float64)
            b = np.array(b_values, dtype=np.float64)

            # Check dimensions
            if A.shape[0] != b.shape[0]:
                messagebox.showerror("Error", "Number of equations does not match number of constants.")
                return

            # Now, call least squares solution function
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during least squares computation: {e}")
                return

            # Display the solution x
            self.display_least_squares_result(x)

    def process_large_file(self):
        try:
            ATA, ATb = process_large_file(self.file_path)
            # Solve the normal equations
            x = solve_normal_equations(ATA, ATb)
            # Display the solution x
            self.display_least_squares_result(x)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during least squares computation: {e}")

    def display_least_squares_result(self, x):
        result_window = tk.Toplevel(self)
        result_window.title("Least Squares Solution")

        # Display solution
        tk.Label(result_window, text="Least Squares Solution:", font=("Arial", 12)).pack(pady=10)
        for i, xi in enumerate(x):
            tk.Label(result_window, text=f"x{i+1} = {xi:.6f}", font=("Arial", 12)).pack()


@njit(parallel=True)
def benchmark_matrix_operations(size):
    # Initialize matrices
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    result = np.zeros((size, size))

    # Count actual FLOPS:
    # Per iteration: 2 sin/cos (15 FLOPS each) + 1 exp (20 FLOPS) + 1 div (1 FLOP) +
    # 2 mul (2 FLOPS) + 2 add (2 FLOPS) + 1 abs (1 FLOP)
    # Total = 56 FLOPS per inner loop iteration
    flops_per_iteration = 56

    for i in prange(size):
        for j in prange(size):
            temp = 0.0
            for k in range(size):
                temp += np.sin(matrix_a[i, k]) * np.cos(matrix_b[k, j])
                temp += np.exp(matrix_a[i, k] * 0.01) / (1.0 + np.abs(matrix_b[k, j]))
            result[i, j] = temp

    total_flops = size * size * size * flops_per_iteration
    return result, total_flops


@njit(parallel=True)
def benchmark_vector_operations(size):
    # Vector operations benchmark
    vector = np.random.rand(size)
    result = np.zeros(size)

    # Count actual FLOPS per iteration:
    # sqrt (15) + abs (1) + log1p (20) + abs (1) +
    # sin (15) + cos (15) + add (1) + mul (1) +
    # exp (20) + abs (1) + mul (1)
    # Total = 91 FLOPS per iteration
    flops_per_iteration = 91

    for i in prange(size):
        result[i] = np.sqrt(np.abs(vector[i])) + np.log1p(np.abs(vector[i]))
        result[i] *= np.sin(vector[i]) + np.cos(vector[i])
        result[i] = np.exp(-np.abs(result[i]))

    total_flops = size * flops_per_iteration
    return result, total_flops


class CpuBenchmarkFp(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("CPU Benchmark Fp")
        self.geometry("800x700")  # Increased height for better display
        self.resizable(True, True)

        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create control frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Benchmark Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Size selection
        ttk.Label(self.control_frame, text="Matrix/Vector Size:").pack(side=tk.LEFT, padx=5)
        self.size_var = tk.StringVar(value="1000")
        self.size_entry = ttk.Entry(self.control_frame, textvariable=self.size_var, width=10)
        self.size_entry.pack(side=tk.LEFT, padx=5)

        # Run button
        self.run_button = ttk.Button(self.control_frame, text="Run Benchmark", command=self.run_benchmark)
        self.run_button.pack(side=tk.LEFT, padx=20)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results text (moved above the plot)
        self.results_text = tk.Text(self.results_frame, height=8, width=50)
        self.results_text.pack(fill=tk.X, padx=5, pady=5)

        # Create figure for plotting
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.benchmark_running = False

    def run_benchmark(self):
        if self.benchmark_running:
            return

        try:
            size = int(self.size_var.get())
            if size <= 0:
                raise ValueError("Size must be positive")
        except ValueError as e:
            tk.messagebox.showerror("Error", f"Invalid size value: {str(e)}")
            return

        self.benchmark_running = True
        self.run_button.config(state='disabled')
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)

        # Create and start benchmark thread
        benchmark_thread = threading.Thread(target=self.run_benchmark_thread, args=(size,))
        benchmark_thread.start()

    def run_benchmark_thread(self, size):
        try:
            # Clear previous plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            # Update progress
            self.progress_var.set(10)
            self.update_idletasks()

            # Warmup run
            self.results_text.insert(tk.END, "Warming up JIT compilation...\n")
            _, _ = benchmark_matrix_operations(100)
            _, _ = benchmark_vector_operations(100)

            self.progress_var.set(30)

            # Matrix operations benchmark
            start_time = time.time()
            _, matrix_flops = benchmark_matrix_operations(size)
            matrix_time = time.time() - start_time

            # Prevent division by zero
            if matrix_time <= 0:
                matrix_time = 1e-9

            self.progress_var.set(60)

            # Vector operations benchmark
            start_time = time.time()
            _, vector_flops = benchmark_vector_operations(size)
            vector_time = time.time() - start_time

            # Prevent division by zero
            if vector_time <= 0:
                vector_time = 1e-9

            self.progress_var.set(80)

            # Calculate GFLOPS
            matrix_gflops = (matrix_flops / matrix_time) / 1e9
            vector_gflops = (vector_flops / vector_time) / 1e9

            # Plot results with exact values
            operations = ['Matrix Operations', 'Vector Operations']
            gflops = [matrix_gflops, vector_gflops]

            # Create bar plot
            bars = ax.bar(operations, gflops)

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

            # Improve y-axis scale
            max_gflops = max(gflops)
            ax.set_ylim(0, max_gflops * 1.2)  # Add 20% padding

            # Format plot
            ax.set_ylabel('GFLOPS (Giga Floating-point Operations Per Second)')
            ax.set_title('Floating Point Performance Benchmark')
            ax.grid(True, linestyle='--', alpha=0.7)

            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=0)

            self.fig.tight_layout()
            self.canvas.draw()

            # Display detailed results
            self.results_text.insert(tk.END, f"\nBenchmark Results (Size: {size}x{size})\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

            self.results_text.insert(tk.END, f"Matrix Operations:\n")
            self.results_text.insert(tk.END, f"  - Time: {matrix_time:.4f} seconds\n")
            self.results_text.insert(tk.END, f"  - Performance: {matrix_gflops:.4f} GFLOPS\n")
            self.results_text.insert(tk.END, f"  - Total FLOPS: {matrix_flops:,}\n\n")

            self.results_text.insert(tk.END, f"Vector Operations:\n")
            self.results_text.insert(tk.END, f"  - Time: {vector_time:.4f} seconds\n")
            self.results_text.insert(tk.END, f"  - Performance: {vector_gflops:.4f} GFLOPS\n")
            self.results_text.insert(tk.END, f"  - Total FLOPS: {vector_flops:,}\n")

            self.progress_var.set(100)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Benchmark failed: {str(e)}")

        finally:
            self.benchmark_running = False
            self.run_button.config(state='normal')


@njit(parallel=True)
def benchmark_integer_matrix_operations(matrix_a, matrix_b):
    size = matrix_a.shape[0]
    result = np.zeros((size, size), dtype=np.int64)
    mod_value = 1000000007  # Large prime for modulo operations
    int_ops_per_iteration = 14

    for i in prange(size):
        for j in prange(size):
            temp = 0
            for k in range(size):
                op1 = (matrix_a[i,k] * matrix_b[k,j]) + (matrix_a[i,k] ^ matrix_b[k,j])
                op1 = op1 % mod_value
                temp += op1  # addition (1)

                op2 = ((matrix_a[i,k] & matrix_b[k,j]) << 2)
                op2 = op2 % mod_value
                temp ^= op2  # bitwise XOR and assignment (1)

                op3 = ((matrix_a[i,k] | matrix_b[k,j]) >> 1)
                op3 = op3 % mod_value
                temp += op3  # addition (1)
            result[i,j] = temp

    total_int_ops = size * size * size * int_ops_per_iteration
    return result, total_int_ops

@njit(parallel=True)
def benchmark_integer_vector_operations(vector):
    size = vector.size
    result = np.zeros(size, dtype=np.int64)
    mod_value = 1000000007
    factor = 123456789
    offset = 987654321
    shift_amount = 5
    mask = 0xFFFFFFFF
    int_ops_per_iteration = 13

    for i in prange(size):
        res = (vector[i] * factor + offset) % mod_value  # 3 ops
        res ^= (vector[i] << shift_amount) & mask        # 3 ops
        res += (vector[i] >> shift_amount) | mask        # 3 ops
        res = res % mod_value                            # 2 ops
        result[i] = res

    total_int_ops = size * int_ops_per_iteration
    return result, total_int_ops

class CpuBenchmarkInteger(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("CPU Integer Benchmark")
        self.geometry("800x700")
        self.resizable(True, True)

        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create control frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Benchmark Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Size selection
        ttk.Label(self.control_frame, text="Matrix/Vector Size:").pack(side=tk.LEFT, padx=5)
        self.size_var = tk.StringVar(value="1000")
        self.size_entry = ttk.Entry(self.control_frame, textvariable=self.size_var, width=10)
        self.size_entry.pack(side=tk.LEFT, padx=5)

        # Run button
        self.run_button = ttk.Button(self.control_frame, text="Run Benchmark", command=self.run_benchmark)
        self.run_button.pack(side=tk.LEFT, padx=20)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results text
        self.results_text = tk.Text(self.results_frame, height=8, width=50)
        self.results_text.pack(fill=tk.X, padx=5, pady=5)

        # Create figure for plotting
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.benchmark_running = False

    def run_benchmark(self):
        if self.benchmark_running:
            return

        try:
            size = int(self.size_var.get())
            if size <= 0:
                raise ValueError("Size must be positive")
        except ValueError as e:
            self.show_error_message(f"Invalid size value: {str(e)}")
            return

        self.benchmark_running = True
        self.run_button.config(state='disabled')
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)

        # Create and start benchmark thread
        benchmark_thread = threading.Thread(target=self.run_benchmark_thread, args=(size,))
        benchmark_thread.start()

    def run_benchmark_thread(self, size):
        try:
            # Generate random matrices and vector
            self.progress_var.set(5)
            self.update_idletasks()

            matrix_a = np.random.randint(1, 1000, size=(size, size), dtype=np.int64)
            matrix_b = np.random.randint(1, 1000, size=(size, size), dtype=np.int64)
            vector = np.random.randint(1, 1000, size=size, dtype=np.int64)

            # Warmup run
            self.results_text.insert(tk.END, "Warming up JIT compilation...\n")
            _, _ = benchmark_integer_matrix_operations(matrix_a[:100, :100], matrix_b[:100, :100])
            _, _ = benchmark_integer_vector_operations(vector[:100])

            self.progress_var.set(30)

            # Integer matrix operations benchmark
            start_time = time.time()
            _, matrix_int_ops = benchmark_integer_matrix_operations(matrix_a, matrix_b)
            matrix_time = time.time() - start_time

            # Prevent division by zero
            if matrix_time <= 0:
                matrix_time = 1e-9

            self.progress_var.set(60)

            # Integer vector operations benchmark
            start_time = time.time()
            _, vector_int_ops = benchmark_integer_vector_operations(vector)
            vector_time = time.time() - start_time

            # Prevent division by zero
            if vector_time <= 0:
                vector_time = 1e-9

            self.progress_var.set(80)

            # Calculate GIOPS (Giga Integer Operations Per Second)
            matrix_giops = (matrix_int_ops / matrix_time) / 1e9
            vector_giops = (vector_int_ops / vector_time) / 1e9

            # Plot results with exact values
            operations = ['Matrix Operations', 'Vector Operations']
            giops = [matrix_giops, vector_giops]

            # Create bar plot
            bars = self.plot_results(operations, giops)

            self.fig.tight_layout()
            self.canvas.draw()

            # Display detailed results
            self.results_text.insert(tk.END, f"\nBenchmark Results (Size: {size}x{size})\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")

            self.results_text.insert(tk.END, f"Integer Matrix Operations:\n")
            self.results_text.insert(tk.END, f"  - Time: {matrix_time:.4f} seconds\n")
            self.results_text.insert(tk.END, f"  - Performance: {matrix_giops:.4f} GIOPS\n")
            self.results_text.insert(tk.END, f"  - Total Integer Operations: {matrix_int_ops:,}\n\n")

            self.results_text.insert(tk.END, f"Integer Vector Operations:\n")
            self.results_text.insert(tk.END, f"  - Time: {vector_time:.4f} seconds\n")
            self.results_text.insert(tk.END, f"  - Performance: {vector_giops:.4f} GIOPS\n")
            self.results_text.insert(tk.END, f"  - Total Integer Operations: {vector_int_ops:,}\n")

            self.progress_var.set(100)

        except Exception as e:
            self.show_error_message(f"Benchmark failed: {str(e)}")

        finally:
            self.benchmark_running = False
            self.run_button.config(state='normal')

    def plot_results(self, operations, giops):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        bars = ax.bar(operations, giops)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        # Improve y-axis scale
        max_giops = max(giops)
        ax.set_ylim(0, max_giops * 1.2)  # Add 20% padding

        # Format plot
        ax.set_ylabel('GIOPS (Giga Integer Operations Per Second)')
        ax.set_title('Integer Performance Benchmark')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=0)
        return bars

    def show_error_message(self, message):
        # Create a new top-level window
        error_window = tk.Toplevel(self)
        error_window.title("Error")
        error_window.geometry("500x300")
        error_window.resizable(True, True)

        # Create a text widget
        text_widget = tk.Text(error_window, wrap='word')
        text_widget.pack(expand=True, fill='both')

        # Insert the error message
        text_widget.insert('1.0', message)

        # Allow text selection and copying
        text_widget.config(state='normal')

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(text_widget, command=text_widget.yview)
        scrollbar.pack(side='right', fill='y')
        text_widget['yscrollcommand'] = scrollbar.set

        # Add a close button
        close_button = ttk.Button(error_window, text="Close", command=error_window.destroy)
        close_button.pack(pady=5)


class Probability_Density_Function(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Probability Density Function")
        self.geometry("600x700")

        # Frame for input
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(pady=10)

        self.create_input_fields()

    def create_input_fields(self):
        # Label and Entry for f(x)
        tk.Label(self.input_frame, text="Enter the probability density function f(x):").grid(row=0, column=0, sticky='e')
        self.function_entry = tk.Entry(self.input_frame, width=50)
        self.function_entry.grid(row=0, column=1)

        # Label and Entry for Lower Limit
        tk.Label(self.input_frame, text="Lower Limit of x (a):").grid(row=1, column=0, sticky='e')
        self.lower_limit_entry = tk.Entry(self.input_frame, width=10)
        self.lower_limit_entry.grid(row=1, column=1, sticky='w')

        # Label and Entry for Upper Limit
        tk.Label(self.input_frame, text="Upper Limit of x (b):").grid(row=2, column=0, sticky='e')
        self.upper_limit_entry = tk.Entry(self.input_frame, width=10)
        self.upper_limit_entry.grid(row=2, column=1, sticky='w')

        # Button to compute expected value
        self.compute_button = tk.Button(self.input_frame, text="Compute Expected Value E(X)", command=self.compute_expected_value)
        self.compute_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Label to display result
        self.result_label = tk.Label(self, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def compute_expected_value(self):
        import numpy as np
        from scipy.integrate import quad
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Get the function string and limits
        function_str = self.function_entry.get()
        lower_limit_str = self.lower_limit_entry.get()
        upper_limit_str = self.upper_limit_entry.get()

        # Validate and process limits
        try:
            lower_limit = self.parse_limit(lower_limit_str)
            upper_limit = self.parse_limit(upper_limit_str)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        if lower_limit >= upper_limit:
            messagebox.showerror("Error", "Lower limit must be less than upper limit.")
            return

        # Prepare the function for evaluation
        try:
            # We define a function of x
            def f(x):
                return eval(function_str, {"x": x, "np": np})
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function. {e}")
            return

        # Compute the expected value E(X) = ∫ x * f(x) dx over [a, b]
        try:
            from scipy.integrate import quad

            # First check that the function is a valid PDF, i.e., ∫ f(x) dx over [a, b] = 1
            total_prob, _ = quad(f, lower_limit, upper_limit, limit=100, epsabs=1e-6)
            if not np.isclose(total_prob, 1.0, atol=1e-4):
                messagebox.showerror("Error",
                                     f"The total probability over the interval [{lower_limit}, {upper_limit}] is {total_prob}, which is not 1. Please ensure the function is a valid PDF.")
                return

            # Now compute the expected value
            def integrand_mean(x):
                return x * f(x)

            expected_value, _ = quad(integrand_mean, lower_limit, upper_limit, limit=100, epsabs=1e-6)

            # Compute E(X^2)
            def integrand_variance(x):
                return x ** 2 * f(x)

            expected_value_squared, _ = quad(integrand_variance, lower_limit, upper_limit, limit=100, epsabs=1e-6)

            # Compute variance and standard deviation
            variance = expected_value_squared - expected_value ** 2
            standard_deviation = np.sqrt(variance)

            # Display the results
            self.result_label.config(
                text=f"E(X) = {expected_value:.4f}\nVar(X) = {variance:.4f}\nσ = {standard_deviation:.4f}")

            # Plot the PDF
            self.plot_pdf(f, lower_limit, upper_limit)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during computation: {e}")
            return

    def parse_limit(self, limit_str):
        import numpy as np

        limit_str = limit_str.strip()
        if limit_str.lower() in ('np.inf', 'inf', '+inf'):
            return np.inf
        elif limit_str.lower() in ('-np.inf', '-inf'):
            return -np.inf
        else:
            try:
                return float(limit_str)
            except ValueError:
                raise ValueError(f"Invalid limit value: {limit_str}")

    def plot_pdf(self, f, lower_limit, upper_limit):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Determine plotting range
        if np.isinf(lower_limit):
            x_min = -10  # Choose a reasonable lower limit for plotting
        else:
            x_min = lower_limit

        if np.isinf(upper_limit):
            x_max = 10  # Choose a reasonable upper limit for plotting
        else:
            x_max = upper_limit

        # Generate x values
        x_values = np.linspace(x_min, x_max, 400)

        # Evaluate f(x) safely
        y_values = []
        for x in x_values:
            try:
                y = f(x)
                y_values.append(y)
            except Exception:
                y_values.append(0)

        y_values = np.array(y_values)

        # Create a new figure
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the PDF
        ax.plot(x_values, y_values, label='f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Probability Density Function')
        ax.legend()

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)




# Functions for processing large files and solving normal equations
def process_large_file(file_path):
    batch_size = 1000  # Adjust based on your system's memory
    num_vars = None

    with open(file_path, 'r') as file:
        # Initialize variables
        ATA = None
        ATb = None
        A_batch = []
        b_batch = []

        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            numbers = line.replace(',', ' ').split()
            if num_vars is None:
                num_vars = len(numbers) - 1
                if num_vars < 1:
                    raise ValueError("Invalid number of variables.")
                # Initialize ATA and ATb
                ATA = np.zeros((num_vars, num_vars), dtype=np.float64)
                ATb = np.zeros(num_vars, dtype=np.float64)
            else:
                if len(numbers) != num_vars + 1:
                    raise ValueError(f"Inconsistent number of variables on line {line_num}.")

            numbers = [float(num) for num in numbers]
            a_i = numbers[:-1]
            b_i = numbers[-1]
            A_batch.append(a_i)
            b_batch.append(b_i)

            # Process batch
            if len(A_batch) == batch_size:
                process_batch(A_batch, b_batch, ATA, ATb)
                A_batch = []
                b_batch = []

        # Process any remaining data
        if A_batch:
            process_batch(A_batch, b_batch, ATA, ATb)

    return ATA, ATb

def process_batch(A_list, b_list, ATA, ATb):
    A_batch = np.array(A_list, dtype=np.float64)
    b_batch = np.array(b_list, dtype=np.float64)

    # Ensure arrays are C-contiguous for BLAS
    A_batch = np.ascontiguousarray(A_batch)
    b_batch = np.ascontiguousarray(b_batch)

    # Update ATA and ATb using BLAS
    ATA_batch = A_batch.T @ A_batch
    ATb_batch = A_batch.T @ b_batch

    ATA += ATA_batch
    ATb += ATb_batch

def solve_normal_equations(ATA, ATb):
    from scipy.linalg import cho_factor, cho_solve

    # Cholesky decomposition
    c, lower = cho_factor(ATA)
    x = cho_solve((c, lower), ATb)
    return x

# Numba-accelerated functions
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def matrix_multiply_numba(A, B):
    return A @ B

@njit(parallel=True, fastmath=True)
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        # Upper Triangular
        for k in prange(i, n):
            sum = 0.0
            for j in range(i):
                sum += L[i, j] * U[j, k]
            U[i, k] = A[i, k] - sum

        # Lower Triangular
        L[i, i] = 1.0
        for k in prange(i+1, n):
            sum = 0.0
            for j in range(i):
                sum += L[k, j] * U[j, i]
            L[k, i] = (A[k, i] - sum) / U[i, i]
    return L, U

@njit(parallel=True, fastmath=True)
def linear_regression(x, y):
    n = x.size
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0

    # Parallel loop with SIMD
    for i in prange(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i] * x[i]
        sum_xy += x[i] * y[i]

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        raise ValueError("Denominator in regression calculation is zero.")
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

@njit(parallel=True, fastmath=True)
def compute_lagrange_polynomial_numba(xi, fi, x_values):
    n = xi.size
    m = x_values.size

    # Compute barycentric weights w_k
    w = np.ones(n)
    for k in range(n):
        for j in range(n):
            if j != k:
                w[k] /= (xi[k] - xi[j])

    y_values = np.empty(m)

    for i in prange(m):
        x = x_values[i]
        numerator = 0.0
        denominator = 0.0
        exact = False

        for k in range(n):
            x_diff = x - xi[k]
            if x_diff == 0.0:
                y_values[i] = fi[k]
                exact = True
                break
            temp = w[k] / x_diff
            numerator += temp * fi[k]
            denominator += temp

        if not exact:
            y_values[i] = numerator / denominator

    return y_values

if __name__ == "__main__":
    app = MatrixApp()
    app.mainloop()
