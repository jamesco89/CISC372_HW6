#fastblur: fastblur.c
#	gcc -g fastblur.c -o fastblur -lm

#clean:
#	rm -f fastblur output.png

cudablur2: cudablur2.cu
	nvcc cudablur2.cu -o cudablur2

clean:
	rm -f cudablur2 output.png
