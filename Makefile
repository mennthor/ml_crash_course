TEXC = lualatex
TEXCOPTS = --output-directory=build


all: part01


part01: build/01-Math.Minimiser.pdf


build/01-Math.Minimiser.pdf: 01-Math.Minimiser.tex | build
	$(TEXC) $(TEXCOPTS) 01-Math.Minimiser.tex


build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
