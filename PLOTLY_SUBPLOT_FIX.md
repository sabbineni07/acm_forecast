# 🔧 Plotly Subplot Fix - Pie Chart Error

## ❌ **The Error:**
```
Trace type 'pie' is not compatible with subplot type 'xy'
at grid position (1, 2)
```

## ✅ **The Solution:**

### **Problem:**
When using `make_subplots()` with pie charts, you need to specify the subplot type as `"domain"` instead of the default `"xy"`.

### **Before (Broken):**
```python
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# This will FAIL:
fig.add_trace(
    go.Pie(labels=categories, values=values),
    row=1, col=2  # Position (1, 2) - ERROR!
)
```

### **After (Fixed):**
```python
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"secondary_y": False}, {"type": "domain"}],  # ← FIXED!
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# This will WORK:
fig.add_trace(
    go.Pie(labels=categories, values=values),
    row=1, col=2  # Position (1, 2) - SUCCESS!
)
```

## 📊 **Subplot Types:**

| Type | Description | Compatible Traces |
|------|-------------|-------------------|
| `"xy"` | Default cartesian plot | Scatter, Bar, Line, etc. |
| `"domain"` | Pie chart area | Pie, Sunburst, Treemap |
| `"polar"` | Polar coordinates | Scatterpolar, Barpolar |
| `"ternary"` | Ternary plot | Scatterternary |
| `"mapbox"` | Map plot | Scattermapbox |
| `"geo"` | Geographic plot | Scattergeo |

## 🎯 **Key Points:**

1. **Pie charts need `"type": "domain"`** in the specs
2. **Position (1, 2)** means row 1, column 2
3. **Other plot types** can use `"secondary_y": False` for xy plots
4. **Mixed subplots** are supported - you can have different types in different positions

## 🚀 **Your Notebook is Now Fixed!**

The `02_data_exploration.ipynb` notebook has been updated and should now run without the pie chart error.

**Test it by running the notebook in Jupyter Lab!** 🎉


