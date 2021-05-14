#fastblur: fastblur.c
#	gcc -g fastblur.c -o fastblur -lm

#clean:
#	rm -f fastblur output.png

CFLAGS=
cudablur2: cudablur2.o
	nvcc $(CFLAGS) cudablur2.o -o cudablur2 -lm

cudablur2.o: cudablur2.cu
	nvcc -c $(CFLAGS) cudablur2.cu -o cudablur2.o

cudablur3: cudanlur3.o
	nvcc $(CFLAGS) cudablur3.o -o cudablur3 -lm

cudablur3.o: cudablur3.cu
	nvcc -c $(CFLAGS) cudablur3.cu -o cudablur3.o

clean:
	rm -f cudablur2 cudablur2.o cudablur3 cudablur3.o output2.png output3.png

