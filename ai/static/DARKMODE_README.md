# Dark Mode Implementation

This directory contains the dark mode functionality for the TransparentSF application.

## Files

- `darkmode.css` - CSS variables and styles for light and dark themes
- `js/darkmode.js` - JavaScript class to handle theme switching and persistence
- `js/add-darkmode.js` - Utility script for quickly adding dark mode to any template

## How to Use

### For Existing Templates

The following templates already have dark mode implemented:
- `backend.html` - Main dashboard with toggle button in header
- `dashboard.html` - Dashboard page
- `anomaly_analyzer.html` - Anomaly analyzer
- `index.html` - Chat interface

### Adding Dark Mode to New Templates

#### Method 1: Manual Implementation (Recommended)

1. Add the CSS and JS files to the `<head>` section:
```html
<link rel="stylesheet" href="/static/darkmode.css">
<script src="/static/js/darkmode.js"></script>
```

2. Update your CSS to use the CSS variables:
```css
body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-primary);
}
```

3. Add a dark mode toggle button (optional):
```html
<button class="dark-mode-toggle" title="Toggle dark mode">
    <span class="icon">🌙</span>
</button>
```

#### Method 2: Quick Implementation

For templates that need minimal changes, you can use the utility script:

```html
<script src="/static/js/add-darkmode.js"></script>
```

This will automatically:
- Load the dark mode CSS and JS files
- Apply basic dark mode styles to common elements
- Initialize the dark mode manager

## CSS Variables

### Light Theme (Default)
- `--bg-primary: #ffffff` - Main background
- `--bg-secondary: #f8f9fa` - Secondary background
- `--bg-tertiary: #ffffff` - Tertiary background
- `--text-primary: #222222` - Primary text
- `--text-secondary: #666666` - Secondary text
- `--text-muted: #999999` - Muted text
- `--border-primary: #E8E9EB` - Primary borders
- `--border-secondary: #f0f0f0` - Secondary borders
- `--bright-purple: #ad35fa` - Brand purple
- `--warm-coral: #FF6B5A` - Brand coral

### Dark Theme
- `--bg-primary: #121212` - Main background
- `--bg-secondary: #1e1e1e` - Secondary background
- `--bg-tertiary: #2a2a2a` - Tertiary background
- `--text-primary: #ffffff` - Primary text
- `--text-secondary: #cccccc` - Secondary text
- `--text-muted: #888888` - Muted text
- `--border-primary: #2a2a2a` - Primary borders
- `--border-secondary: #404040` - Secondary borders
- `--bright-purple: #c44dff` - Brand purple (adjusted for dark)
- `--warm-coral: #FF6B5A` - Brand coral (unchanged)

## JavaScript API

The dark mode functionality is managed by the `DarkModeManager` class:

```javascript
// Access the global instance
const darkMode = window.darkModeManager;

// Toggle theme
darkMode.toggleTheme();

// Set specific theme
darkMode.setTheme('dark'); // or 'light'

// Get current theme
const currentTheme = darkMode.currentTheme;

// Listen for theme changes
window.addEventListener('themeChanged', (event) => {
    console.log('Theme changed to:', event.detail.theme);
});
```

## Features

- **Theme Persistence**: Theme preference is saved in localStorage
- **Cross-tab Synchronization**: Theme changes sync across browser tabs
- **Iframe Communication**: Theme changes propagate to iframes
- **Smooth Transitions**: CSS transitions for theme switching
- **Automatic Icon Updates**: Toggle button icons update automatically

## Browser Support

- Modern browsers with CSS custom properties support
- localStorage for theme persistence
- postMessage API for iframe communication

## Troubleshooting

### Theme not applying
1. Check that the CSS and JS files are loaded
2. Verify CSS variables are being used instead of hardcoded colors
3. Check browser console for JavaScript errors

### Toggle button not working
1. Ensure the button has the `dark-mode-toggle` class
2. Check that the JavaScript is loaded after the DOM
3. Verify the button structure includes the icon span

### Iframe theme not syncing
1. Check that both parent and iframe are on the same domain
2. Verify postMessage is not being blocked
3. Ensure the iframe has the dark mode scripts loaded 