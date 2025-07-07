// Utility script to quickly add dark mode to any template
// Usage: Include this script in the <head> section of any template

(function() {
    'use strict';
    
    // Add the dark mode CSS and JS files if they don't exist
    function addDarkModeFiles() {
        // Check if dark mode CSS is already loaded
        if (!document.querySelector('link[href="/static/darkmode.css"]')) {
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = '/static/darkmode.css';
            document.head.appendChild(cssLink);
        }
        
        // Check if dark mode JS is already loaded
        if (!document.querySelector('script[src="/static/js/darkmode.js"]')) {
            const jsScript = document.createElement('script');
            jsScript.src = '/static/js/darkmode.js';
            document.head.appendChild(jsScript);
        }
    }
    
    // Apply dark mode styles to common elements
    function applyDarkModeStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* Quick dark mode overrides for common elements */
            body {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Common container elements */
            .container, .main, .content, .wrapper {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Common card/box elements */
            .card, .box, .panel, .section {
                background-color: var(--bg-primary) !important;
                border-color: var(--border-primary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Form elements */
            input, textarea, select {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-primary) !important;
            }
            
            /* Tables */
            table {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            th, td {
                border-color: var(--border-primary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Buttons */
            button {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-primary) !important;
            }
            
            /* Links */
            a {
                color: var(--bright-purple) !important;
            }
            
            a:hover {
                color: var(--warm-coral) !important;
            }
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
            }
            
            /* Paragraphs and text */
            p, span, div {
                color: var(--text-primary) !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Initialize dark mode
    function initDarkMode() {
        addDarkModeFiles();
        applyDarkModeStyles();
        
        // Wait for dark mode manager to be available
        const checkDarkMode = setInterval(() => {
            if (window.darkModeManager) {
                clearInterval(checkDarkMode);
                console.log('Dark mode initialized successfully');
            }
        }, 100);
    }
    
    // Run initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDarkMode);
    } else {
        initDarkMode();
    }
    
    // Export for manual use
    window.quickDarkMode = {
        init: initDarkMode,
        addFiles: addDarkModeFiles,
        applyStyles: applyDarkModeStyles
    };
    
})(); 