TEXC = lualatex
TEXCOPTS = --output-directory=build --halt-on-error
TEXDEPS = header.tex

TARGET01 = 01-Math.Minimizer.Stats

all: $(TARGET01)


$(TARGET01): build/$(TARGET01).pdf


build/$(TARGET01).pdf: $(TARGET01).tex $(TEXDEPS) | build
	$(TEXC) $(TEXCOPTS) $(TARGET01).tex


build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
