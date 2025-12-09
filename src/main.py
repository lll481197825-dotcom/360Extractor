#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from core.settings_manager import SettingsManager

def main():
    app = QApplication(sys.argv)
    
    # Initialize settings
    SettingsManager()

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()