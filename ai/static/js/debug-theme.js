// Debug theme state
// Run this in the browser console to see current theme state

(function() {
    'use strict';
    
    console.log('=== Theme Debug Info ===');
    
    // Check localStorage
    const storedTheme = localStorage.getItem('theme');
    console.log('localStorage theme:', storedTheme);
    
    // Check document attribute
    const docTheme = document.documentElement.getAttribute('data-theme');
    console.log('document data-theme:', docTheme);
    
    // Check if dark mode manager exists
    if (window.darkModeManager) {
        console.log('Dark mode manager current theme:', window.darkModeManager.currentTheme);
        console.log('Dark mode manager initialized:', window.darkModeManager.isInitialized);
    } else {
        console.log('Dark mode manager not found');
    }
    
    // Check if we're in an iframe
    const isInIframe = window.parent !== window;
    console.log('In iframe:', isInIframe);
    
    // Check computed styles
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    console.log('Body background color:', bodyBg);
    
    // Check toggle button
    const toggle = document.querySelector('.dark-mode-toggle');
    if (toggle) {
        const icon = toggle.querySelector('.icon');
        console.log('Toggle button icon:', icon ? icon.textContent : 'not found');
    } else {
        console.log('Toggle button not found');
    }
    
    // Check system preference
    if (window.matchMedia) {
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        console.log('System prefers dark mode:', systemPrefersDark);
    }
    
    // Check if iframe communication is working
    if (isInIframe) {
        console.log('Attempting to request theme from parent...');
        try {
            window.parent.postMessage({
                type: 'requestTheme'
            }, '*');
            console.log('Theme request sent to parent');
        } catch (e) {
            console.log('Failed to send theme request to parent:', e);
        }
    }
    
    // Check if we have iframes and can communicate with them
    const iframes = document.querySelectorAll('iframe');
    console.log('Number of iframes on page:', iframes.length);
    
    console.log('=== End Debug Info ===');
    
})(); 