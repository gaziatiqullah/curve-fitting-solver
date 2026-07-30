import numpy as np
from typing import List, Tuple, Dict

def parse_data(data_str: str) -> List[Tuple[float, float]]:
    """Parse input data string to list of (x, y) tuples"""
    points = []
    for line in data_str.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
            except ValueError:
                continue
    return points

def linear_regression(points: List[Tuple[float, float]]) -> Dict:
    """Linear regression: y = mx + c with precise HTML formatted steps"""
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_x2 = sum(p[0] ** 2 for p in points)
    
    # Matching screenshot formulas:
    denom = (n * sum_x2) - (sum_x ** 2)
    m_num = (n * sum_xy) - (sum_x * sum_y)
    c_num = (sum_x2 * sum_y) - (sum_x * sum_xy)
    
    m = m_num / denom if denom != 0 else 0
    c = c_num / denom if denom != 0 else 0
    
    # 1. HTML Table Generation
    table_rows = "".join([
        f"<tr><td>{p[0]:g}</td><td>{p[1]:g}</td><td>{p[0]**2:g}</td><td>{p[0]*p[1]:g}</td></tr>"
        for p in points
    ])
    
    step1_html = f"""
    <div class="step-header">Step 1: Given data points (n = {n})</div>
    <div class="table-wrapper">
        <table class="calc-table">
            <thead>
                <tr>
                    <th>x</th>
                    <th>y</th>
                    <th>x²</th>
                    <th>xy</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
            <tfoot>
                <tr>
                    <td>Σx = {sum_x:g}</td>
                    <td>Σy = {sum_y:g}</td>
                    <td>Σx² = {sum_x2:g}</td>
                    <td>Σxy = {sum_xy:g}</td>
                </tr>
            </tfoot>
        </table>
    </div>
    """

    # 2. Normal Equations Box Generation
    step2_html = f"""
    <div class="step-header">Step 2: Normal Equations (Least Squares)</div>
    <div class="box-equations">
        Σy = n·c + m·Σx &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ {sum_y:g} = {n}·c + {sum_x:g}·m<br>
        Σxy = c·Σx + m·Σx² &nbsp;&nbsp;→ {sum_xy:g} = {sum_x:g}·c + {sum_x2:g}·m
    </div>
    """

    # 3. Final Solution Box Generation (matching your math)
    step3_html = f"""
    <div class="step-header">Step 3: Solve the system</div>
    <div class="box-solution">
        m = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²) = {m_num:g} / {denom:g} = <span class="highlight-val">{m:g}</span><br>
        c = (Σx²·Σy - Σx·Σxy) / (n·Σx² - (Σx)²) = {c_num:g} / {denom:g} = <span class="highlight-val">{c:g}</span>
    </div>
    """

    full_steps_html = step1_html + step2_html + step3_html
    
    sign = "+" if c >= 0 else "-"
    equation = f"y = {m:g}x {sign} {abs(c):g}"
    
    def predict(x: float) -> float:
        return m * x + c
    
    # Generate graph points
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    return {
        "method": "linear",
        "equation": equation,
        "coefficients": {"m": m, "c": c},
        "steps": full_steps_html,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def polynomial_regression(points: List[Tuple[float, float]], degree: int) -> Dict:
    """Polynomial regression formatted step outputs"""
    n = len(points)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    N = degree + 1
    A = [[0.0] * N for _ in range(N)]
    B = [0.0] * N
    
    for i in range(N):
        for j in range(N):
            A[i][j] = sum(x ** (i + j) for x in x_vals)
        B[i] = sum((x ** i) * y for x, y in points)
    
    coeffs = gauss_elimination(A, B)
    
    terms = []
    for i, coef in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coef:.4f}")
        elif i == 1:
            terms.append(f"{coef:.4f}x")
        else:
            terms.append(f"{coef:.4f}x^{i}")
    equation = "y = " + " + ".join(terms).replace("+ -", "- ")
    
    def predict(x: float) -> float:
        return sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
    
    min_x = min(x_vals)
    max_x = max(x_vals)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    coeff_str = "<br>".join([f"a<sub>{i}</sub> = <span class='highlight-val'>{c:.4f}</span>" for i, c in enumerate(coeffs)])
    steps_html = f"""
    <div class="step-header">Polynomial Regression (Degree {degree})</div>
    <div class="box-solution">
        <strong>Data points (n):</strong> {n}<br><br>
        <strong>Calculated Coefficients:</strong><br>
        {coeff_str}
    </div>
    """
    
    return {
        "method": "polynomial",
        "equation": equation,
        "coefficients": coeffs,
        "steps": steps_html,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def exponential_regression(points: List[Tuple[float, float]]) -> Dict:
    """Exponential regression formatted step outputs"""
    transformed = [(x, np.log(y)) for x, y in points if y > 0]
    n = len(transformed)
    
    sum_x = sum(p[0] for p in transformed)
    sum_y = sum(p[1] for p in transformed)
    sum_xy = sum(p[0] * p[1] for p in transformed)
    sum_x2 = sum(p[0] ** 2 for p in transformed)
    
    det = n * sum_x2 - sum_x ** 2
    b = (n * sum_xy - sum_x * sum_y) / det
    A = (sum_x2 * sum_y - sum_x * sum_xy) / det
    a = np.exp(A)
    
    equation = f"y = {a:.4f}·e^({b:.4f}x)"
    
    def predict(x: float) -> float:
        return a * np.exp(b * x)
    
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    steps_html = f"""
    <div class="step-header">Exponential Regression: y = a · e^(bx)</div>
    <div class="box-equations">
        Linearized Form: ln(y) = ln(a) + b·x
    </div>
    <div class="step-header">Solution</div>
    <div class="box-solution">
        a = <span class="highlight-val">{a:.4f}</span><br>
        b = <span class="highlight-val">{b:.4f}</span>
    </div>
    """
    
    return {
        "method": "exponential",
        "equation": equation,
        "coefficients": {"a": a, "b": b},
        "steps": steps_html,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def logarithmic_regression(points: List[Tuple[float, float]]) -> Dict:
    """Logarithmic regression formatted step outputs"""
    valid = [(x, y) for x, y in points if x > 0]
    n = len(valid)
    
    transformed = [(np.log(x), y) for x, y in valid]
    
    sum_x = sum(p[0] for p in transformed)
    sum_y = sum(p[1] for p in transformed)
    sum_xy = sum(p[0] * p[1] for p in transformed)
    sum_x2 = sum(p[0] ** 2 for p in transformed)
    
    det = n * sum_x2 - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / det
    b = (sum_x2 * sum_y - sum_x * sum_xy) / det
    
    equation = f"y = {a:.4f}·ln(x) + {b:.4f}"
    
    def predict(x: float) -> float:
        return a * np.log(x) + b
    
    min_x = min(p[0] for p in valid)
    max_x = max(p[0] for p in valid)
    graph_x = np.linspace(min_x - 0.1*(max_x-min_x), max_x + 0.1*(max_x-min_x), 100)
    graph_y = [predict(x) for x in graph_x]
    
    steps_html = f"""
    <div class="step-header">Logarithmic Regression: y = a · ln(x) + b</div>
    <div class="box-equations">
        Linearized Form using X = ln(x): y = a·X + b
    </div>
    <div class="step-header">Solution</div>
    <div class="box-solution">
        a = <span class="highlight-val">{a:.4f}</span><br>
        b = <span class="highlight-val">{b:.4f}</span>
    </div>
    """
    
    return {
        "method": "logarithmic",
        "equation": equation,
        "coefficients": {"a": a, "b": b},
        "steps": steps_html,
        "predict": predict,
        "graph": {"x": graph_x.tolist(), "y": graph_y}
    }

def gauss_elimination(A: List[List[float]], B: List[float]) -> List[float]:
    """Solve Ax = B using Gaussian elimination with partial pivoting"""
    n = len(A)
    a = [row[:] for row in A]
    b = B[:]
    
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(a[k][i]) > abs(a[max_row][i]):
                max_row = k
        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        for k in range(i + 1, n):
            factor = a[k][i] / a[i][i]
            b[k] -= factor * b[i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
                
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / a[i][i]
        for k in range(i - 1, -1, -1):
            b[k] -= a[k][i] * x[i]
            
    return x