@ECHO OFF

REM Command file for building Sphinx documentation

IF "%SPHINXBUILD%" == "" (
    SET SPHINXBUILD=sphinx-build
)
SET BUILDDIR=build
SET SOURCEDIR=source
SET ALLSPHINXOPTS=-d %BUILDDIR%/doctrees %SPHINXOPTS% %SOURCEDIR%
SET I18NSPHINXOPTS=%SPHINXOPTS% %SOURCEDIR%

IF NOT "%PAPER%" == "" (
    SET ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
    SET I18NSPHINXOPTS=-D latex_paper_size=%PAPER% %I18NSPHINXOPTS%
)

IF "%1" == "" GOTO help

IF "%1" == "help" (
:help
    ECHO Please use `make ^<target^>` where ^<target^> is one of
    ECHO   html       to make standalone HTML files
    ECHO   dirhtml    to make HTML files named index.html in directories
    ECHO   singlehtml to make a single large HTML file
    ECHO   pickle     to make pickle files
    ECHO   json       to make JSON files
    ECHO   htmlhelp   to make HTML files and a HTML help project
    ECHO   qthelp     to make HTML files and a qthelp project
    ECHO   devhelp    to make HTML files and a Devhelp project
    ECHO   epub       to make an epub
    ECHO   latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter
    ECHO   text       to make text files
    ECHO   man        to make manual pages
    ECHO   texinfo    to make Texinfo files
    ECHO   gettext    to make PO message catalogs
    ECHO   changes    to make an overview of all changed/added/deprecated items
    ECHO   linkcheck  to check all external links for integrity
    ECHO   doctest    to run all doctests embedded in the documentation
    GOTO end
)

IF "%1" == "clean" (
    FOR /D %%i IN (%BUILDDIR%\*) DO RMDIR /Q /S %%i
    DEL /Q /S %BUILDDIR%\*
    GOTO end
)

IF "%1" == "html" (
    %SPHINXBUILD% -b html %ALLSPHINXOPTS% %BUILDDIR%/html
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The HTML pages are in %BUILDDIR%/html.
    GOTO end
)

IF "%1" == "dirhtml" (
    %SPHINXBUILD% -b dirhtml %ALLSPHINXOPTS% %BUILDDIR%/dirhtml
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The HTML pages are in %BUILDDIR%/dirhtml.
    GOTO end
)

IF "%1" == "singlehtml" (
    %SPHINXBUILD% -b singlehtml %ALLSPHINXOPTS% %BUILDDIR%/singlehtml
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The HTML page is in %BUILDDIR%/singlehtml.
    GOTO end
)

IF "%1" == "pickle" (
    %SPHINXBUILD% -b pickle %ALLSPHINXOPTS% %BUILDDIR%/pickle
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished; now you can process the pickle files.
    GOTO end
)

IF "%1" == "json" (
    %SPHINXBUILD% -b json %ALLSPHINXOPTS% %BUILDDIR%/json
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished; now you can process the JSON files.
    GOTO end
)

IF "%1" == "htmlhelp" (
    %SPHINXBUILD% -b htmlhelp %ALLSPHINXOPTS% %BUILDDIR%/htmlhelp
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The HTML Help project is in %BUILDDIR%/htmlhelp.
    GOTO end
)

IF "%1" == "qthelp" (
    %SPHINXBUILD% -b qthelp %ALLSPHINXOPTS% %BUILDDIR%/qthelp
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The QtHelp files are in %BUILDDIR%/qthelp.
    GOTO end
)

IF "%1" == "devhelp" (
    %SPHINXBUILD% -b devhelp %ALLSPHINXOPTS% %BUILDDIR%/devhelp
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The Devhelp files are in %BUILDDIR%/devhelp.
    GOTO end
)

IF "%1" == "epub" (
    %SPHINXBUILD% -b epub %ALLSPHINXOPTS% %BUILDDIR%/epub
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The EPUB file is in %BUILDDIR%/epub.
    GOTO end
)

IF "%1" == "latex" (
    %SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%/latex
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The LaTeX files are in %BUILDDIR%/latex.
    GOTO end
)

IF "%1" == "text" (
    %SPHINXBUILD% -b text %ALLSPHINXOPTS% %BUILDDIR%/text
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The text files are in %BUILDDIR%/text.
    GOTO end
)

IF "%1" == "man" (
    %SPHINXBUILD% -b man %ALLSPHINXOPTS% %BUILDDIR%/man
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The manual pages are in %BUILDDIR%/man.
    GOTO end
)

IF "%1" == "texinfo" (
    %SPHINXBUILD% -b texinfo %ALLSPHINXOPTS% %BUILDDIR%/texinfo
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The Texinfo files are in %BUILDDIR%/texinfo.
    GOTO end
)

IF "%1" == "gettext" (
    %SPHINXBUILD% -b gettext %I18NSPHINXOPTS% %BUILDDIR%/locale
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Build finished. The message catalogs are in %BUILDDIR%/locale.
    GOTO end
)

IF "%1" == "changes" (
    %SPHINXBUILD% -b changes %ALLSPHINXOPTS% %BUILDDIR%/changes
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO The overview file is in %BUILDDIR%/changes.
    GOTO end
)

IF "%1" == "linkcheck" (
    %SPHINXBUILD% -b linkcheck %ALLSPHINXOPTS% %BUILDDIR%/linkcheck
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Link check complete. Check %BUILDDIR%/linkcheck/output.txt for errors.
    GOTO end
)

IF "%1" == "doctest" (
    %SPHINXBUILD% -b doctest %ALLSPHINXOPTS% %BUILDDIR%/doctest
    IF ERRORLEVEL 1 EXIT /B 1
    ECHO.
    ECHO Doctests complete. See %BUILDDIR%/doctest/output.txt.
    GOTO end
)

:end
