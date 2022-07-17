#include <cstdio>
#include <map>
#include <algorithm>
#include <iostream>
using namespace std;
int main()
{
    char a = '\0';
    map<char, int> cmap1;
    map<char, int> cmap2;
    // printf("cmap1\n");
    while((a=getchar())!='\n') 
    {
        // scanf("%c", &a); 
        map<char, int>::iterator it = cmap1.find(a);
        if (it != cmap1.end())
        {
            
            // int temp = it->second;
            // // printf("temp %d ", temp);
            // temp++;
            // cmap1.insert(pair<char, int>(a, temp));
            cmap1[a]++;
        }
        else
        {
            cmap1.insert(pair<char, int>(a, 1));
        }
        
    } 
    // printf("cmap2\n");
    while((a=getchar())!='\n') 
    {
        // scanf("%c", &a); 
        map<char, int>::iterator it = cmap2.find(a);
        if (it != cmap2.end())
        {
            // int temp = it->second;
            // temp++;
            // printf("temp %d ", temp);

            cmap2[a]++;
        }
        else
        {
            cmap2.insert(pair<char, int>(a, 1));
        }
        
    } 

    for (map<char, int>::iterator it1=cmap1.begin(); it1 != cmap1.end(); it1++)
    {
        printf("%c %d ", it1->first, it1->second);
    }
    printf("\n");
    for (map<char, int>::iterator it1=cmap2.begin(); it1 != cmap2.end(); it1++)
    {
        printf("%c %d ", it1->first, it1->second);
    }
    if (cmap1 == cmap2)
    {
        printf("YES\n");
    }
    else
    {
        printf("NO\n");
    }
}