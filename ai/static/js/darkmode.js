// Dark Mode JavaScript functionality

class DarkModeManager {
    constructor() {
        this.currentTheme = this.getStoredTheme() || 'light';
        this.isToggling = false; // Prevent rapid toggling
        this.isInitialized = false; // Prevent double initialization
        this.init();
    }

    init() {
        // Prevent double initialization
        if (this.isInitialized) {
            return;
        }
        
        // Set initial theme
        this.setTheme(this.currentTheme, false); // Don't notify iframes on init
        
        // Add event listeners for theme changes
        this.setupEventListeners();
        
        // Mark as initialized
        this.isInitialized = true;
        
        // Now notify iframes after initialization
        setTimeout(() => {
            this.notifyIframes();
        }, 100);
        
        console.log('Dark mode manager initialized with theme:', this.currentTheme);
    }

    getStoredTheme() {
        const stored = localStorage.getItem('theme');
        
        // If no theme is stored, check system preference
        if (!stored) {
            const systemPrefersDark = this.detectSystemDarkMode();
            const defaultTheme = systemPrefersDark ? 'dark' : 'light';
            localStorage.setItem('theme', defaultTheme);
            return defaultTheme;
        }
        
        // Return the stored theme (don't reset dark to light)
        return stored;
    }

    detectSystemDarkMode() {
        // Check if the user's system prefers dark mode
        // This uses the prefers-color-scheme media query
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    setStoredTheme(theme) {
        localStorage.setItem('theme', theme);
    }

    setTheme(theme, notifyIframes = true) {
        // Prevent setting the same theme multiple times
        if (this.currentTheme === theme && this.isInitialized) {
            return;
        }
        
        this.currentTheme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        this.setStoredTheme(theme);
        
        // Update toggle button if it exists
        this.updateToggleButton();
        
        // Notify iframes only if requested
        if (notifyIframes) {
            this.notifyIframes();
        }
        
        // Dispatch custom event for other scripts
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
    }

    toggleTheme() {
        // Prevent rapid toggling
        if (this.isToggling) {
            return;
        }
        
        this.isToggling = true;
        
        // Add visual feedback
        const toggle = document.querySelector('.dark-mode-toggle');
        if (toggle) {
            toggle.style.transform = 'scale(0.9)';
            toggle.style.opacity = '0.7';
        }
        
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
        
        // Reset visual feedback and allow another toggle
        setTimeout(() => {
            if (toggle) {
                toggle.style.transform = '';
                toggle.style.opacity = '';
            }
            this.isToggling = false;
        }, 300);
    }

    updateToggleButton() {
        const toggle = document.querySelector('.dark-mode-toggle');
        if (toggle) {
            const icon = toggle.querySelector('.icon');
            if (icon) {
                icon.textContent = this.currentTheme === 'light' ? '🌙' : '☀️';
            }
        }
    }

    setupEventListeners() {
        // Use a more reliable event delegation approach
        document.addEventListener('click', (e) => {
            // Check if the clicked element or its parent is the toggle button
            const toggleButton = e.target.closest('.dark-mode-toggle');
            if (toggleButton) {
                e.preventDefault();
                e.stopPropagation();
                this.toggleTheme();
            }
        });

        // Also add direct event listeners to existing toggle buttons
        this.addDirectEventListeners();

        // Listen for theme changes from parent window (if in iframe)
        window.addEventListener('message', (e) => {
            if (e.data && e.data.type === 'themeChange') {
                // Only update if the theme is different to prevent loops
                if (e.data.theme !== this.currentTheme) {
                    this.setTheme(e.data.theme, false); // Don't notify back to prevent loops
                }
            }
        });

        // Listen for storage changes (for cross-tab synchronization)
        window.addEventListener('storage', (e) => {
            if (e.key === 'theme' && e.newValue) {
                // Only update if the theme is different to prevent loops
                if (e.newValue !== this.currentTheme) {
                    this.setTheme(e.newValue, false); // Don't notify iframes to prevent loops
                }
            }
        });

        // Listen for system theme changes
        this.setupSystemThemeListener();

        // Watch for new toggle buttons being added to the DOM
        this.observeNewButtons();
    }

    addDirectEventListeners() {
        // Add direct click listeners to existing toggle buttons
        const toggleButtons = document.querySelectorAll('.dark-mode-toggle');
        toggleButtons.forEach(button => {
            // Remove any existing listeners to prevent duplicates
            button.removeEventListener('click', this.handleToggleClick);
            // Add new listener
            button.addEventListener('click', this.handleToggleClick.bind(this));
        });
    }

    handleToggleClick(e) {
        e.preventDefault();
        e.stopPropagation();
        this.toggleTheme();
    }

    observeNewButtons() {
        // Use MutationObserver to watch for new toggle buttons
        const observer = new MutationObserver((mutations) => {
            let shouldUpdate = false;
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            if (node.classList && node.classList.contains('dark-mode-toggle')) {
                                shouldUpdate = true;
                            } else if (node.querySelector && node.querySelector('.dark-mode-toggle')) {
                                shouldUpdate = true;
                            }
                        }
                    });
                }
            });
            
            if (shouldUpdate) {
                this.addDirectEventListeners();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    notifyIframes() {
        // Notify all iframes in the current page
        const iframes = document.querySelectorAll('iframe');
        iframes.forEach(iframe => {
            try {
                // Send theme multiple times to ensure it's received
                const sendTheme = () => {
                    iframe.contentWindow.postMessage({
                        type: 'themeChange',
                        theme: this.currentTheme
                    }, '*');
                };
                
                // Send immediately
                sendTheme();
                
                // Send again after a short delay to ensure iframe is ready
                setTimeout(sendTheme, 100);
                setTimeout(sendTheme, 500);
                setTimeout(sendTheme, 1000);
                
            } catch (e) {
                // Ignore cross-origin errors
            }
        });

        // If we're in an iframe, notify the parent
        if (window.parent !== window) {
            try {
                window.parent.postMessage({
                    type: 'themeChange',
                    theme: this.currentTheme
                }, '*');
            } catch (e) {
                // Ignore cross-origin errors
            }
        }
    }

    // Method to create and add the toggle button
    createToggleButton(container) {
        const toggle = document.createElement('button');
        toggle.className = 'dark-mode-toggle';
        toggle.title = 'Toggle dark mode';
        toggle.innerHTML = '<span class="icon">🌙</span>';
        
        // Add direct event listener
        toggle.addEventListener('click', this.handleToggleClick.bind(this));
        
        // Update the icon based on current theme
        this.updateToggleButton();
        
        if (container) {
            container.appendChild(toggle);
        }
        
        return toggle;
    }

    // Method to reset theme to system preference
    resetToSystemPreference() {
        localStorage.removeItem('theme');
        const systemPrefersDark = this.detectSystemDarkMode();
        const systemTheme = systemPrefersDark ? 'dark' : 'light';
        this.setTheme(systemTheme);
    }

    // Method to reset theme to light mode (legacy method)
    resetToLightMode() {
        localStorage.setItem('theme', 'light');
        this.setTheme('light');
    }

    setupSystemThemeListener() {
        // Check if the browser supports the media query
        if (window.matchMedia) {
            const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
            
            // Add listener for system theme changes
            const handleSystemThemeChange = (e) => {
                // Only auto-update if user hasn't manually set a preference
                const stored = localStorage.getItem('theme');
                if (!stored) {
                    const newTheme = e.matches ? 'dark' : 'light';
                    this.setTheme(newTheme);
                    console.log('System theme changed to:', newTheme);
                }
            };
            
            // Add the listener
            if (mediaQuery.addEventListener) {
                mediaQuery.addEventListener('change', handleSystemThemeChange);
            } else {
                // Fallback for older browsers
                mediaQuery.addListener(handleSystemThemeChange);
            }
            
            // Also check initial system preference if no theme is stored
            const stored = localStorage.getItem('theme');
            if (!stored) {
                const systemPrefersDark = mediaQuery.matches;
                const defaultTheme = systemPrefersDark ? 'dark' : 'light';
                this.setTheme(defaultTheme);
                console.log('Initial system theme applied:', defaultTheme);
            }
        }
    }
}

// Initialize dark mode manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.darkModeManager = new DarkModeManager();
});

// Also initialize immediately if DOM is already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.darkModeManager = new DarkModeManager();
    });
} else {
    window.darkModeManager = new DarkModeManager();
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DarkModeManager;
} 