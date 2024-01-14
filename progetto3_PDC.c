/*************************************************************************************
 Studente: Pastore Luca
 Matricola: N97000431

 Anno Accademico: 2023/24
 Corso: Parallel And Distributed Computing
 Prof.: Laccetti Giuliano, Mele Valeria
 Corso di Laurea Magistrale in Informatica
 Università degli studi Federico II - Napoli

 Traccia:
    Sviluppare un algoritmo per il calcolo del prodotto matrice-matrice, in ambiente di
    calcolo parallelo su architettura MIMD a memoria distribuita, che utilizzi la libreria MPI.

    Le matrici A di mxm elementi e B di mxm elementi devono essere distribuite a pxp processi (per
    valori di m multipli di p), disposti secondo una topologia a griglia bidimensionale.

    L'algoritmo deve implementare la strategia di comunicazione BMR ed essere organizzato in modo
    da utilizzare un sottoprogramma per la costruzione della griglia bidimensionale dei processi
    ed un sottoprogramma per il calcolo dei prodotti locali.
**************************************************************************************/

// SEZIONE DICHIARAZIONE HEADER FILE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"

/*
INPUT DA TERMINALE
l'input è costituito da un valore passati via terminale
    - argv[1] = numero di righe/colonne della matrice (la matrice è quadrata)
N.B.: argv[0] contiene il nome del programma
    - argc = contiene il numero di parametri di input passati da terminale
*/

// SEZIONE DICHIARAZIONE PROTOTIPI FUNZIONI
int perfect_square_checker(int x);
int input_check(int argc, char **argv, int nproc);
double *create_matrix(int n);
void print_matrix(double *A, int n, char *matrix_name);
void create_processors_grid(MPI_Comm *grid_communicator, MPI_Comm *grid_communicator_row, MPI_Comm *grid_communicator_column, int menum, int nproc, int grid_coordinates_dim, int grid_dimension, int *coordinate);
void send_matrix(double *matrix, int matrix_dimension, int submatrix_dim, int nproc, MPI_Comm *grid_communicator, int coords_dim);
void receive_matrix(double *submatrix, int submatrix_row, int submatrix_columns, int menum, MPI_Comm *grid_communicator, int coords_dim, int is_for_A);
void BMR(int matrix_dim, double *sub_matr_A, double *sub_matr_B, int grid_dimension, int grid_menum, int grid_coordinates_dim, int *coordinates, MPI_Comm *grid_communicator);
double *matrix_product(double *sub_matr_A, double *sub_matr_B, double *sub_matr_C, int rows_subA, int columns_subA, int columns_subB);
double *receive_result_matrix(double *C, int matrix_dim, int grid_dim, MPI_Comm *comm_grid);
void send_result_matrix(double *C, int matrix_dim, int grid_dim, int grid_menum, MPI_Comm *comm_grid);

// INIZIO MAIN PROGRAMM
int main(int argc, char **argv){
    // dichiarazioni variabili
    MPI_Status status;
    MPI_Comm    grid_communicator, 
                grid_communicator_row, 
                grid_communicator_column;

    int menum,                // identificatore processo
        nproc,                // numero di processi
        matr_dim,             // numero di righe/colonne matrice
        grid_dimension,       // dimensione della griglia di processori
        grid_coordinates_dim, // dimensione array delle coordinate
        *coordinates,         // array contenente le coordinate del processo nella topologia a griglia
        input_checked,        //variabile contenente informazioni sulla correttezza dell'input'
        block_dimension, tag, i, j, r, grid_menum, p, q;

    double *matrix_A,  // matrice A di valori double di cui fare il prodotto
        *matrix_B,     // matrice B di valori double di cui fare il prodotto
        *matrix_ris,   // matrice che conterrà il risultato del prodotto tra A e B
        *sub_matr_A,   // sottomatrice locale A
        *sub_matr_B,   // sottomatrice locale B
        *sub_matr_ris, // sottomatrice locale risultato
        start_time,    // timestamp iniziale per calcolo tempo di esecuzione
        end_time,      // timestamp finale per calcolo tempo di esecuzione
        delta_time,    // differenza tra i due timestamp che da il tempo di computazione totale
        max_time;

    srand(time(NULL));

    // inizializzazione funzioni MPI
    MPI_Init(&argc, &argv); // inizializzazione ambiente MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &menum); // salvo identificativo processo
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // salvo numero di processi totale

    if (menum == 0){ // operazioni svolte dal processori 0
        // verifica dell'input
        input_checked = input_check(argc, argv, nproc);
        if (input_checked != 0)
            MPI_Abort(MPI_COMM_WORLD, input_checked);

        matr_dim = atoi(argv[1]); // salvataggio della dimensione delle matrici quadrate A e B

        grid_dimension = sqrt(nproc); // calcolo le dimensioni della griglia di processori, date dalla radice quadrata del numero di processori

        block_dimension = matr_dim / grid_dimension; // computazione dimensione del blocco
    }

    // Distribuzione dei dati di input ai vari processi
    MPI_Bcast(&grid_dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matr_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //inizializzazione matrici A e B, allocazione memoria per matrice risultato
    if(menum == 0){
        matrix_A = create_matrix(matr_dim); // allocazione memoria e inizializzazione matrice A
        matrix_B = create_matrix(matr_dim); // allocazione memoria e inizializzazione matrice B
        matrix_ris = (double *)malloc(matr_dim * matr_dim * sizeof(double)); // allocazione memoria matrice risultato
        //print_matrix(matrix_A, matr_dim, "Matrice A:"); // stampa matrice A appena generata
        //print_matrix(matrix_B, matr_dim, "Matrice B:"); // stampa matrice B appena generata
    } 

    if( nproc > 1 ){ // La strategia BRM e la creazione della topologia a griglia viene applicata solo se i processi attivi sono piu di 1
        // creazione topologia a griglia di processori
        grid_coordinates_dim = 2; // La grigia di processori sarà in due dimensioni, si hanno due valori per le coordinate di ciascun processore
        coordinates = (int *)malloc(grid_coordinates_dim * sizeof(int));
        create_processors_grid(&grid_communicator, &grid_communicator_row, &grid_communicator_column, menum, nproc, grid_coordinates_dim, grid_dimension, coordinates);
        MPI_Comm_rank(grid_communicator, &grid_menum);

        // Allocazione memoria per sottomatrici locali
        sub_matr_A = (double *)malloc((matr_dim / grid_dimension) * matr_dim * sizeof(double));
        sub_matr_B = (double *)malloc(matr_dim * (matr_dim / grid_dimension) * sizeof(double));
        sub_matr_ris = (double *)malloc((matr_dim / grid_dimension) * (matr_dim / grid_dimension) * sizeof(double));
                            
        // Distribuzione delle sottomatrici a ciascun processo, operazione svolta dal processo con id 0
        if( menum == 0){
            send_matrix(matrix_A, matr_dim, matr_dim / grid_dimension, nproc, &grid_communicator, grid_coordinates_dim); // Invio della sottomatrice di A
            send_matrix(matrix_B, matr_dim, matr_dim / grid_dimension, nproc, &grid_communicator, grid_coordinates_dim); // Invio della sottomatrice di B

            // Distribuzione dei valori di sub_matr_A
            for (i = 0; i < matr_dim/2; i++)
                for (j = 0; j < matr_dim; j++)
                    sub_matr_A[(matr_dim * i) + j] = matrix_A[(matr_dim * i) + j];

            // Distribuzione dei valori di sub_matr_B
            r = 0;
            for (i = 0; i < matr_dim; i++)
                for (j = 0; j < matr_dim/grid_dimension; j++)
                    sub_matr_B[r++] = matrix_B[(matr_dim * i) + j];
        } else {
            // Ricezione delle sottomatrici per ciascun processo con id != 0
            receive_matrix(sub_matr_A, matr_dim / grid_dimension, matr_dim, menum, &grid_communicator, grid_coordinates_dim, 1); // Ricezione sottomatrice di A
            receive_matrix(sub_matr_B, matr_dim, matr_dim / grid_dimension, menum, &grid_communicator, grid_coordinates_dim, 0); // Ricezione sottomatrice di B
        }

        // Inizio misurazione tempo di esecuzione del prodotto matrice - matrice
        MPI_Barrier(MPI_COMM_WORLD); // Barriera di sincronizzazione per la misurazione del timestamp iniziale
        start_time = MPI_Wtime(); // Salvataggio timestamp iniziale

        // Applicazione strategia di comunicazione BMR
        BMR(matr_dim, sub_matr_A, sub_matr_B, grid_dimension, grid_menum, grid_coordinates_dim, coordinates, &grid_communicator);

        // Computo del prodotto tra matrice A e B - quest'operazione viene effettuata per ogni processore
        sub_matr_ris = matrix_product(sub_matr_A, sub_matr_B, sub_matr_ris, matr_dim / grid_dimension, matr_dim, matr_dim / grid_dimension);

        // Ricezione dei prodotti locali da parte del processore con id = 0
        if (menum == 0){
            matrix_ris = receive_result_matrix(matrix_ris, matr_dim, grid_dimension, &grid_communicator);

            // Assegnazione del proprio sub_matr_ris in matrix_ris
            p = 0;  q = 0;

            for (i = 0; i < block_dimension; i++, q = q + (matr_dim - block_dimension))
                for (j = 0; j < block_dimension; j++)
                    matrix_ris[q++] = sub_matr_ris[p++];

        } else { 
            // Invio dei prodotti locali al processore con id = 0 da parte di tutti gli altri processori
            send_result_matrix(sub_matr_ris, matr_dim, grid_dimension, grid_menum, &grid_communicator);
        }

        // Fine misurazione tempo di calcolo del prodotto matrice-matrice
        end_time = MPI_Wtime(); // Salvataggio timestamp finale
        delta_time = end_time - start_time; // calcolo del tempo totale
        MPI_Reduce(&delta_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }else{
        // Esecuzione del prodotto nel caso di un unico processore attivo
        start_time = MPI_Wtime(); // Salvataggio timestamp iniziale
        matrix_product(matrix_A, matrix_B, matrix_ris, matr_dim, matr_dim, matr_dim); // calcolo del prodotto
        end_time = MPI_Wtime(); // Salvataggio timestamp finale
        max_time = end_time - start_time; // calcolo del tempo totale
    }

    // Stampa del risultato da parte del processore 0
    if (menum == 0){
        // Stampa della matrice risultato
        //print_matrix(matrix_ris, matr_dim, "Matrice Risultato:");

        // Stampa del tempo totale di calcolo
        printf(" Dimensione matrice: %d x %d \n\n Tempo impiegato: %lf secondi\n\n", matr_dim, matr_dim, max_time);
    }

    // terminazione MPI
    MPI_Finalize();

    return 0;
} // FINE MAIN PROGRAMM

// SEZIONE DEFINIZIONE FUNZIONI E PROCEDURE

/*
    Funzione utilizzata per verificare che un valore x sia un quadrato perfetto o meno
    - x = valore da verificare
    output: 0 se x è un quadrato perfetto, -1 altrimenti
*/
int perfect_square_checker(int x){
    int int_root_part = (int)sqrt(x);
    double dec_root_part = sqrt(x) - int_root_part;

    return (dec_root_part == 0.0) ? 0 : -1;
}

/*
    Funzione utilizzata per controllare la correttezza dei parametri passati in input
    - argc = numero di parametri passati in input
    - argc = array contenente le stringhe passate in input 
    - nproc = numero di processori utilizzati
    output: 
        0 se l'input è corretto
        1 se il numero di parametri passati in input è diversi da 1
        2 se il numero di processi non rende possibile creare una topologia a griglia
        3 se la dimensione n della matrice non è un multiplo della dimensione della griglia di processi
    
    N.B. argv conterrà nella prima posizione il nome del programma, argc conterà per tanto un parametro in più
*/
int input_check(int argc, char **argv, int nproc){
    if(argc != 2){ // verifica correttezza numero di parametri
        printf("\n\n ERRORE ! Inserire come parametro di input solamente il numero di righe/colonne della matrice\n\n");
        return 1;
    }

    // verifica  che il numero di processori sia un quadrato perfetto. al fine di creare una topologia a griglia quadrata
    if(perfect_square_checker(nproc) < 0){ 
        printf("\n\n ERRORE ! Non e' possibile creare una topologia a griglia utilizzando %d processi\n\n", nproc);
        return 2;
    }

    int matrix_dim = atoi(argv[1]);
    if (matrix_dim % nproc != 0 || matrix_dim <= 0){ // verifica che la dimensione della matrice sia multiplo del numero di processori
        printf("\n\n ERRORE ! La dimensione della matrice deve essere un multiplo della dimensione della griglia\n\n");
        return 3;
    }

    return 0; // verifiche di correttezza superate
}

// SEZINE DEFINIZIONE FUNZIONI

/*
    Funzione utilizzata per creare una matrice di valori double di dimensioni n*n
    - n = numero di righe e di colonne della matrice
    output: matrice di double di n*n elementi
*/
double *create_matrix(int n){
    double *A; // Matrice n * n
    int i, j;

    // FASE DI ALLOCAZIONE MEMORIA
    A = (double *)malloc(n * n * sizeof(double)); // allocazione di n * n celle di memoria per matrice 

    // FASE DI INIZIALIZZAZIONE CASUALE DELLA MATRICE
    for (i = 0; i < n; i++) 
        for (j = 0; j < n; j++)
            A[(i * n) + j] = rand() % 50; // generazione casuale dei valori della matrice

    return A;
}

/*
    Funzione utilizzata per stampare i valori di una matrice n*n di double
    - A = matrice di valori double
    - n = numero di righe/colonne di A
    - matrix_name = nome della matrice da visualizzare a video
*/
void print_matrix(double *A, int n, char *matrix_name){
    int i, j;

    printf("\n Matrice %s :\n", matrix_name);
    for (i = 0; i < n; i++){
        printf("\n");
        for (j = 0; j < n; j++)
            printf(" %.2lf\t", A[(i * n) + j]);
    } 
    printf("\n\n");
}

/*
    Funzione utilizzata per creare una topologia a griglia di processori
    - menum = id processore
    - nproc = numero di processori totale
    - grid_coordinates_dim = dimensione array delle coordinate
    - grid_dimension = dimensione della griglia di processori da creare
    - coordinate = vettore delle coordinate di un processore nella griglia
*/
void create_processors_grid(MPI_Comm *grid_communicator, MPI_Comm *grid_communicator_row, MPI_Comm *grid_communicator_column, int menum, int nproc, int grid_coordinates_dim, int grid_dimension, int *coordinate){
    int *dimensions, // Array per i valori delle dimensioni della griglia
        *period, // Array per la periodicità della griglia
        vc[2], reorder;

    // Inizializzazione array dimension
    dimensions = (int *)malloc(grid_coordinates_dim * sizeof(int));
    dimensions[0] = grid_dimension;
    dimensions[1] = grid_dimension;

    // Inizializzazione array period
    period = (int *)malloc(grid_coordinates_dim * sizeof(int));
    period[0] = 1;
    period[1] = 1;

    reorder = 0;

    // Creazione topologia a griglia 2D di processori
    MPI_Cart_create(MPI_COMM_WORLD, grid_coordinates_dim, dimensions, period, reorder, grid_communicator);

    // Assegnazione delle coordinate a ciascun processore
    MPI_Cart_coords(*grid_communicator, menum, grid_coordinates_dim, coordinate);

    // Creazione subcommunicator di riga
    vc[0] = 0;  vc[1] = 1;
    MPI_Cart_sub(*grid_communicator, vc, grid_communicator_row);

    // Creazione subcommunicator di colonna
    vc[0] = 1;  vc[1] = 0;
    MPI_Cart_sub(*grid_communicator, vc, grid_communicator_column);
}

/*
    Funzione utilizzata per inviare le sottomatrici di A e B ai vari processori (richiamata dal processore con id 0)
    - matrix = matrice da distribuire
    - matrix_dimension = dimensione della matrice
    - submatrix_dim = dimensione della sottomatrice 
    - nproc = numero totale di processi
    - grid_communicator = MPI comunicator di griglia
    - coords_dim = dimensione array delle coordinate
*/
void send_matrix(double *matrix, int matrix_dimension, int submatrix_dim, int nproc, MPI_Comm *grid_communicator, int coords_dim){
    int *local_coords;
    int start_row, start_col, tag, processor, r;

    local_coords = (int *)malloc(coords_dim * sizeof(int));

    for (processor = 1; processor < nproc; processor++){
        
        // Ottenimento delle coordinate della topologia a griglia del processore corrente
        MPI_Cart_coords(*grid_communicator, processor, coords_dim, local_coords);

        // Calcolo degli indici di partenza della matrice
        start_row = local_coords[0] * submatrix_dim;
        start_col = local_coords[1] * submatrix_dim;

        // Invio di submatrix_dim righe
        for (r = 0; r < submatrix_dim; r++){
            tag = 22 + processor;
            MPI_Send(&matrix[(start_row + r) * matrix_dimension + start_col], submatrix_dim, MPI_DOUBLE, processor, tag, MPI_COMM_WORLD);
        }
    }
}

/*
    Funzione utilizzata per la ricezione delle sottomatrici di A e B per i vari processori
    - submatrix = sottomatrice da ricevere
    - submatrix_row = numero di righe della sottomatrice da ricevere
    - submatrix_columns = numero di colonne della sottomatrice da ricevere
    - menum = id del processore attuale
    - grid_communicator = MPI comunicator di griglia
    - coords_dim = dimensione array delle coordinate
    - is_for_A = flag che indica se si sta ricevendo la matrice A o B (1 = ricezione matrice A / 0 = ricezione matrice B)
*/
void receive_matrix(double *submatrix, int submatrix_row, int submatrix_columns, int menum, MPI_Comm *grid_communicator, int coords_dim, int is_for_A){
    int tag, start, i;
    int *local_coords;
    MPI_Status status;

    local_coords = (int *)malloc(coords_dim * sizeof(int));

    /*
    Utilizzo di MPI_Cart_coord per ottenere le coordinate del processori cosi da ottenere il punto
    iniziale per inserire i valori nella sottomatrice.
    */
    MPI_Cart_coords(*grid_communicator, menum, coords_dim, local_coords);
    tag = 22 + menum;

    if (is_for_A == 1){
        start = local_coords[1] * submatrix_row;
        for (i = 0; i < submatrix_row; i++)
            MPI_Recv(&submatrix[i * submatrix_columns + start], submatrix_row, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    } else {
        start = local_coords[0] * submatrix_columns * submatrix_columns;
        for (i = 0; i < submatrix_columns; i++)
            MPI_Recv(&submatrix[i * submatrix_columns + start], submatrix_columns, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    }
}

/*
    Funzione che implementa la strategia di comunicazione BMR
    - matrix_dim = dimensione delle matrici quadrate A e B
    - sub_matr_A = sotto matrice A
    - sub_matr_B = sotto matrice B
    - grid_dimension = dimensione della griglia di processori (numero di colonne/righe)
    - grid_menum = identificativo del processore richiamante
    - grid_coordinates_dim = dimensione del vettore coordinate 
    - coordinates = array contenente le coordinate dei processori nella griglia
    - grid_communicator = MPI comunicator di griglia
*/
void BMR(int matrix_dim, double *sub_matr_A, double *sub_matr_B, int grid_dimension, int grid_menum, int grid_coordinates_dim, int *coordinates, MPI_Comm *grid_communicator){
    MPI_Status status;

    int starting_processor, ending_processor,
        block_dim, *local_coordinates,
        x, y, i, tag;

    block_dim = matrix_dim / grid_dimension; // calcolo dimensione del blocco per sottomatrice
    local_coordinates = (int *)malloc(grid_coordinates_dim * sizeof(int));

    /******* Sottomatrice A *******/
    // Calcolo l'intervallo dei processori che si trovano sulla stessa riga del processore attuale
    starting_processor = grid_menum - coordinates[1];
    ending_processor = grid_dimension * (coordinates[0] + 1) - 1;
    
    // Per ogni processore sulla stessa riga invio il sub_matr_A e ricevo il sub_matr_A
    for (x = starting_processor; x <= ending_processor; x++){
        if (x != grid_menum){
            
            // Invio dei valori di sub_matr_A in blocchi di righe
            for (i = 0; i < block_dim; i++){
                tag = x + 22;
                MPI_Send(&sub_matr_A[coordinates[1] * block_dim + (matrix_dim * i)], block_dim, MPI_DOUBLE, x, tag, MPI_COMM_WORLD);
            }
            
            // Ricezione dei valori di sub_matr_A
            for (i = 0; i < block_dim; i++){
                // Determinazione del processore da cui ricevere i valori di A
                MPI_Cart_coords(*grid_communicator, x, grid_coordinates_dim, local_coordinates); // Ricezione delle coordinate del processore da cui ricevere
                tag = grid_menum + 22;
                MPI_Recv(&sub_matr_A[local_coordinates[1] * block_dim + (matrix_dim * i)], block_dim, MPI_DOUBLE, x, tag, MPI_COMM_WORLD, &status);
            }
        }
    }

    /******* Sottomatrice B *******/
    // Calcolo l'intervallo dei processori che si trovano sulla stessa colonna del processore attuale
    starting_processor = grid_menum - coordinates[0] * grid_dimension;
    ending_processor = grid_dimension * (grid_dimension - 1) + coordinates[1];

    for (y = starting_processor; y <= ending_processor; y += grid_dimension){
        if (y != grid_menum){
            // Invio i valori di sub_matr_B in blocchi di righe
            for (i = 0; i < block_dim; i++){
                tag = y + 22;
                MPI_Send(&sub_matr_B[coordinates[0] * block_dim * block_dim + (block_dim * i)], block_dim, MPI_DOUBLE, y, tag, MPI_COMM_WORLD);
            }
            
            // Ricezione dei valori di sub_matr_B
            for (i = 0; i < block_dim; i++){
                // Determinazione del processore da cui ricevere i valori di B
                MPI_Cart_coords(*grid_communicator, y, grid_coordinates_dim, local_coordinates); // Ricezione delle coordinate del processore da cui ricevere
                tag = grid_menum + 22; 
                MPI_Recv(&sub_matr_B[local_coordinates[0] * block_dim * block_dim + (block_dim * i)], block_dim, MPI_DOUBLE, y, tag, MPI_COMM_WORLD, &status);
            }
        }
    }
}

/*
    Funzione utilizzata per effettuare il prodotto riga-colonna tra due matrici
    - sub_matr_A = matrice A
    - sub_matr_B = matrice B
    - sub_matr_C = matrice che conterrà il risultato
    - rows_subA = numero righe per matrice A
    - columns_subA = numero di colonne per matrice A
    - columns_subB = numero di colonne per matrice B
*/
double *matrix_product(double *sub_matr_A, double *sub_matr_B, double *sub_matr_C, int rows_subA, int columns_subA, int columns_subB){
    int i, j, k, l, a_ij, b_kj, i_c, matrix_dim;
    
    matrix_dim = columns_subA;

    for (i = 0; i < rows_subA; i++){ // Iterazione su righe della matrice A

        for (j = 0; j < columns_subB; j++){ // Iterazione su colonne della matrice B
            sub_matr_C[(i * columns_subB) + j] = 0;
            
            // Computo del prodotto scalare tra i valori della riga di sub_matr_A ed i valori della colonna di sub_matr_B
            for (k = 0; k < matrix_dim; k++){
                i_c = (i * columns_subB) + j;

                a_ij = sub_matr_A[(i * columns_subA) + k];
                b_kj = sub_matr_B[(k * columns_subB) + j];
                sub_matr_C[i_c] = sub_matr_C[i_c] + (a_ij * b_kj);

            }
        }
    }

    return sub_matr_C;
}

/*
    Funzione utilizzata per ricevere le sottomatrici locali a computazione terminata.
    La funzione viene richiamata dal processore con identificativo 0
    - C = matrice contenente il risultato
    - matrix_dim = dimensione della matrice C
    - grid_dim = dimensione della griglia di processori
    - comm_grid = comunicator MPI di griglia
*/
double *receive_result_matrix(double *C, int matrix_dim, int grid_dim, MPI_Comm *comm_grid){
    MPI_Status status;

    int i, j, curr_row, start, tag, block_dim, coords_dim, *local_coords;

    block_dim = matrix_dim / grid_dim; // computo delle dimensioni del blocco da ricevere

    coords_dim = 2;
    local_coords = (int *)malloc(coords_dim * sizeof(int));

    // Ricezione delle matrici locali da tutti i processori con id != 0
    for (i = 1; i < grid_dim * grid_dim; i++){
        tag = 60 + i;

        MPI_Cart_coords(*comm_grid, i, coords_dim, local_coords);
        start = local_coords[0] * matrix_dim * block_dim + local_coords[1] * block_dim;

        for (curr_row = 0; curr_row < block_dim; curr_row++, start = start + matrix_dim)
            MPI_Recv(&C[start], block_dim, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
    }

    return C;
}

/*
    Funzione utilizzata per inviare la sottomatrice locale, a computazione finita, dal processore richiamante
    al processore con id 0
    - C = matrice da inviare
    - matrix_dim = dimensione della matrice
    - grid_dim = dimensione della griglia di processori
    - grid_menum = id del processore chiamante
    - comm_grid = comunicator MPI di griglia
*/
void send_result_matrix(double *C, int matrix_dim, int grid_dim, int grid_menum, MPI_Comm *comm_grid){
    int i, tag, block_dim;
    
    tag = 60;
    block_dim = matrix_dim / grid_dim; //computo delle dimensioni del blocco da inviare

    // Invio della sottomatrice completa
    for (i = 0; i < block_dim; i++)
        MPI_Send(&C[block_dim * i], block_dim, MPI_DOUBLE, 0, tag + grid_menum, MPI_COMM_WORLD);
}