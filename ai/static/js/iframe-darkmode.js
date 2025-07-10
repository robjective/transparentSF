// Dark Mode for iframe content
// This script automatically applies dark mode to iframe content

(function() {
    'use strict';
    
    // Check if we're in an iframe
    const isInIframe = window.parent !== window;
    
    // Function to apply theme
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        
        // Update any toggle buttons if they exist
        const toggles = document.querySelectorAll('.dark-mode-toggle');
        toggles.forEach(toggle => {
            const icon = toggle.querySelector('.icon');
            if (icon) {
                icon.textContent = theme === 'light' ? '🌙' : '☀️';
            }
        });
        
        // Update logo based on theme
        const logoImages = document.querySelectorAll('#wordmark-logo, img[src*="wordmark.png"]');
        logoImages.forEach(img => {
            if (theme === 'dark') {
                // Use dark wordmark for dark mode
                img.src = '/static/darkwordmark.png';
            } else {
                // Use regular wordmark for light mode
                img.src = '/static/wordmark.png';
            }
        });
        
        console.log('Theme applied to iframe:', theme);
    }
    
    // Function to load dark mode CSS and JS
    function loadDarkMode() {
        // Load CSS if not already loaded
        if (!document.querySelector('link[href="/static/darkmode.css"]')) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/static/darkmode.css';
            document.head.appendChild(cssLink);
        }
        
        // Load JS if not already loaded
        if (!window.darkModeManager) {
            const jsScript = document.createElement('script');
            jsScript.src = '/static/js/darkmode.js';
            document.head.appendChild(jsScript);
        }
    }
    
    // Listen for theme changes from parent window
    window.addEventListener('message', (e) => {
        if (e.data && e.data.type === 'themeChange') {
            console.log('Received theme change from parent:', e.data.theme);
            applyTheme(e.data.theme);
        }
    });
    
    // Request current theme from parent
    if (isInIframe) {
        // Request theme immediately
        try {
            window.parent.postMessage({
                type: 'requestTheme'
            }, '*');
        } catch (e) {
            // Ignore cross-origin errors
        }
        
        // Also request after a delay to ensure parent is ready
        setTimeout(() => {
            try {
                window.parent.postMessage({
                    type: 'requestTheme'
                }, '*');
            } catch (e) {
                // Ignore cross-origin errors
            }
        }, 100);
        
        // And again after a longer delay
        setTimeout(() => {
            try {
                window.parent.postMessage({
                    type: 'requestTheme'
                }, '*');
            } catch (e) {
                // Ignore cross-origin errors
            }
        }, 500);
    }
    
    // Load dark mode resources
    loadDarkMode();
    
    // Apply theme based on localStorage (fallback)
    const storedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(storedTheme);
    
    console.log('Iframe dark mode initialized');
    
})(); 