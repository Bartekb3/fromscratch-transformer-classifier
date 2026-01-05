Najwazniejsze zadania:

Napisac streszczenie polskie i angielskie, podsumowanie pracy (ostatni chapter)

Przeczytac całość - poprawic wszystkie bledy - zgodnie z ponizszym README, uzypelnic fragmenty \TODO{}

Tekst powinien byc w zwięzły - jesli mozna usunac jakies zbedne slowa bez zmiany sensu zdania - zrob to

Jak juz wszystko jest napisane wtedy sprawdzic pojedyncze znaki na koncu linii (niełamliwa spacja), ew. poprawic pozycjonowania table, wykresow

Znalezc wiecej prac do bibliografii 



README:

wszystkie angielskie słowa zastąpić polskimi - czesto powtarzaja sie atencja, checkpoint, finetuning-> uwaga, zapisany stan modelu, dostrajanie. W eksperymentach accuracy -> dokladnosc itp.

wszystkie tabele i rysunki powinny miec pozycjonowanie t! ewentualnie b!, chyba ze koniecznie musi byc w srodku tekstu wtedy h!


ja staralem sie nie uzywac recznego łamania linii 

we wszystkich tabelach i rysunkach upewnic sie ze jest label i pomiedzy caption a tabelą jest /smallskip 
- do wszystkich tabel i rysunkow musi byc odwolanie w tekście 

wszystkie angielkie tlumaczanie w formacie (ang. \emph{english translation}) - nie uzywam wszedzie tylko przy pierwszym wystapieniu ale czasami warto przypomniec tlumaczenie 

nie uzywac pogrubionego tekstu w pracy - jedynie przy np.listach na poczatku definicji itemu

Wszystkie liczby powinny byc w trybie matematycznym wewnatrz  $$ chyba jedynie z wyjatkiem tych w tabeli parametrow - config


Wszystkie rownania ktore nie sa czescia zdania powinny byc ponumerowane (chyba zrobione) to znaczy używamy \begin{equation} dla pojedynczego równania ,
\begin{align} jesli chcemy mieć kilka równań pod soba wszystkie ponumerowane , \begin{equation}
\begin{aligned} - kilka równań ale jeden numer




-----------

0. Uzywamy komendy \TODO{ todo } - zeby dodac czerwony komentarz 

1. Pojedyncze znaki na końcu wiersza: Bolek i~Lolek

2. W języku polskim separatorem części ułamkowej liczb jest przecinek. Nie piszemy jednak: $1,5$, lecz tak: $1{,}5$. Wszystkie liczby zapisujemy w trybie matematycznym $$ -



4.referencja do równań używamy \eqref{etykieta}


Format tabeli: ew. z tabularx

\begin{table}[t!] %t! zalecane
\centering
\caption{Tytuł}
\label{tab:etykieta}
\smallskip
\small
\begin{tabular}{@{}ll@{}} %ta linia w zależności od potrzeb
\toprule
\textbf{col 1} & \textbf{col2}\\
\midrule
    & text & text \\
\midrule
    & text & text  \\
\bottomrule
\end{tabular}
\end{table}

6. 
Wszystkie ilustracje labelujemy, tytuł dajemy na DOLE, nie stawiamy kropki po tytule, odwołujemy się w tekście 

7.	Rozdział 1, Podrozdział 1.1, Punkt 1.1.1, Rysunek 7, Tablica 2 – tak piszemy na początku zdania (oczywiście numeracja jest wstawiana automatycznie z \ref{...}). W środku zdania skracamy: zob. rozdz. 1, podrozdz. 1.1, p. 1.1.1, rys. 2, tab. 8.

8. 	Cytat:  ,,text''
	wielokroper: \dots
	wtrącenie, zakres liczbowy za pomocą: --
	pojedynczy myslnik - do złożeń



10. Wykresy w formacie pdf lub svg



12. Tego używamy:
\begin{theorem} - Twierdzenie
\begin{lemma} - Lemat
\begin{example} - Przykład
\begin{proposition} - Stwierdzenie
\begin{corollary} - Wniosek
\begin{definition} - Definicja 
\begin{remark} - Uwaga

13. Układ pracy:

	•	streszczenie pracy, wybrać odpowiednio:
– streszczenie pracy w języku polskim, zawierające tytuł pracy, zestaw słów kluczowych, (objętość 1 strona, odstęp pojedynczy, czcionka 12) – streszczenie pracy w języku angielskim, zawierające tytuł pracy, zestaw słów kluczowych, (objętość 1 strona, odstęp pojedynczy, czcionka 12)

	•	strona tytułowa 
	•	spis treści
	•	w pracy dyplomowej zespołowej: opis podziału prac obejmujący zakres wkładu każdego ze współautorów w część praktyczną (zespołowy projekt programistyczny) oraz część opisową pracy (dla kierunku Inżynieria i Analiza Danych/Data Science – zobacz szablon wkładu współautorów zamieszczony w sekcji „Zalecenia dodatkowe” poniżej)
	•	dla kierunku Inżynieria i Analiza Danych/Data Science: oświadczenie o (nie)używaniu narzędzi SI – zobacz szablon wkładu współautorów zamieszczony w sekcji „Zalecenia dodatkowe” poniżej
	•	kolejne rozdziały pracy
	•	bibliografię
	•	wykaz symboli i skrótów
	•	spis rysunków
	•	spis tabel
	•	spis załączników
	•	załączniki










