#CFLAGS=
#fastblur: obj/fastblur.o
#	gcc $(CFLAGS) obj/fastblur.o -o fastblur -lm


#obj/fastblur.o: fastblur.c
#	gcc -c $(CFLAGS) fastblur.c -o obj/fastblur.o 

fastblur: fastblur.c
	gcc -g  fastblur.c -o fastblur -lm

clean:
	rm -f fastblur output.png
