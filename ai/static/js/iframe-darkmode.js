// Dark Mode for all chart content
// This script applies dark mode based on CLIENT preference only
// (same behavior regardless of embedding context or domain)

(function() {
    'use strict';
    
    // Prevent conflicts with main darkmode.js if both are loaded
    if (window.darkModeManager && window.location === window.parent.location) {
        // We're in the main window with darkmode.js loaded, let it handle everything
        console.log('iframe-darkmode.js: Main darkmode.js detected, deferring to it');
        return;
    }
    
    let isSystemThemeListenerAdded = false;
    let currentTheme = null;
    
    // Function to apply theme
    function applyTheme(theme) {
        if (currentTheme === theme) {
            return; // Already applied, avoid unnecessary updates
        }
        
        currentTheme = theme;
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
        
        // Dispatch theme change event for other scripts (like chart updates)
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
        
        console.log('iframe-darkmode.js: Theme applied:', theme);
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
        
        // Only load main darkmode.js if we're not in an iframe and it's not already loaded
        if (window.location === window.parent.location && !window.darkModeManager) {
            const jsScript = document.createElement('script');
            jsScript.src = '/static/js/darkmode.js';
            document.head.appendChild(jsScript);
        }
    }
    
    // Get client's theme preference - always check system first for embedded content
    function getClientTheme() {
        // For embedded content, always check system preference first
        const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Check stored theme preference
        let stored = null;
        try {
            stored = localStorage.getItem('theme');
        } catch (e) {
            // localStorage might not be available in some embedding contexts
            console.log('iframe-darkmode.js: localStorage not available, using system preference');
        }
        
        // If no theme is stored, use system preference
        if (!stored) {
            const defaultTheme = systemPrefersDark ? 'dark' : 'light';
            try {
                localStorage.setItem('theme', defaultTheme);
            } catch (e) {
                // Ignore if localStorage is not available
            }
            return defaultTheme;
        }
        
        return stored;
    }
    
    // Function to add system theme change listener
    function addSystemThemeListener() {
        if (isSystemThemeListenerAdded) {
            return; // Already added
        }
        
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            
            const handleSystemThemeChange = (e) => {
                console.log('iframe-darkmode.js: System theme change detected:', e.matches ? 'dark' : 'light');
                
                // Always follow system changes for embedded content
                const newTheme = e.matches ? 'dark' : 'light';
                
                // Only update if the theme actually changed
                if (currentTheme !== newTheme) {
                    applyTheme(newTheme);
                    try {
                        localStorage.setItem('theme', newTheme);
                    } catch (e) {
                        // Ignore if localStorage is not available
                    }
                    console.log('iframe-darkmode.js: Chart updated to system theme:', newTheme);
                }
            };
            
            // Add the listener
            if (mediaQuery.addEventListener) {
                mediaQuery.addEventListener('change', handleSystemThemeChange);
                console.log('iframe-darkmode.js: System theme change listener added (modern)');
            } else {
                // Fallback for older browsers
                mediaQuery.addListener(handleSystemThemeChange);
                console.log('iframe-darkmode.js: System theme change listener added (legacy)');
            }
            
            isSystemThemeListenerAdded = true;
            
            // Test the current system preference
            console.log('iframe-darkmode.js: Current system prefers dark:', mediaQuery.matches);
        } else {
            console.log('iframe-darkmode.js: matchMedia not supported');
        }
    }
    
    // Initialize dark mode
    function initializeDarkMode() {
        // Load dark mode resources
        loadDarkMode();
        
        // Get and apply client's theme preference
        const clientTheme = getClientTheme();
        applyTheme(clientTheme);
        console.log('iframe-darkmode.js: Applied initial client theme:', clientTheme);
        
        // Add system theme change listener
        addSystemThemeListener();
        
        console.log('iframe-darkmode.js: Initialization complete');
    }
    
    // Wait for DOM to be ready, then initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeDarkMode);
    } else {
        // DOM is already ready
        initializeDarkMode();
    }
    
    // Also initialize immediately if we're in an embedded context and DOM is ready
    if (window.location !== window.parent.location && document.readyState !== 'loading') {
        initializeDarkMode();
    }
    
    console.log('iframe-darkmode.js: Script loaded and ready');
    
})(); 