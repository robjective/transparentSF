// Check dark mode support
// Run this in the browser console to see which templates have dark mode

(function() {
    'use strict';
    
    console.log('=== Dark Mode Support Check ===');
    
    // Check if dark mode CSS is loaded
    const darkModeCSS = document.querySelector('link[href="/static/darkmode.css"]');
    console.log('Dark mode CSS loaded:', !!darkModeCSS);
    
    // Check if dark mode JS is loaded
    const darkModeJS = document.querySelector('script[src="/static/js/darkmode.js"]');
    console.log('Dark mode JS loaded:', !!darkModeJS);
    
    // Check if iframe dark mode JS is loaded
    const iframeDarkModeJS = document.querySelector('script[src="/static/js/iframe-darkmode.js"]');
    console.log('Iframe dark mode JS loaded:', !!iframeDarkModeJS);
    
    // Check if dark mode manager exists
    console.log('Dark mode manager exists:', !!window.darkModeManager);
    
    // Check current theme
    const currentTheme = document.documentElement.getAttribute('data-theme');
    console.log('Current theme attribute:', currentTheme);
    
    // Check if we're in an iframe
    const isInIframe = window.parent !== window;
    console.log('In iframe:', isInIframe);
    
    // Check if toggle button exists
    const toggleButton = document.querySelector('.dark-mode-toggle');
    console.log('Toggle button exists:', !!toggleButton);
    
    // Check localStorage
    const storedTheme = localStorage.getItem('theme');
    console.log('localStorage theme:', storedTheme);
    
    // Check computed styles
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    console.log('Body background color:', bodyBg);
    
    // Recommendations
    console.log('\n=== Recommendations ===');
    
    if (!darkModeCSS) {
        console.log('❌ Add: <link rel="stylesheet" href="/static/darkmode.css">');
    }
    
    if (!darkModeJS) {
        console.log('❌ Add: <script src="/static/js/darkmode.js"></script>');
    }
    
    if (isInIframe && !iframeDarkModeJS) {
        console.log('❌ Add: <script src="/static/js/iframe-darkmode.js"></script>');
    }
    
    if (!window.darkModeManager) {
        console.log('❌ Dark mode manager not initialized');
    }
    
    if (!currentTheme) {
        console.log('❌ No theme attribute set on document.documentElement');
    }
    
    console.log('=== End Check ===');
    
})(); 