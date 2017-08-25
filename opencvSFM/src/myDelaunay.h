class MyDelauny : public Subdiv2D
{
    // skips "outer" triangles.
    void indices(vector<int> &ind) const
    {    
        int i, total = (int)(qedges.size()*4);
        vector<bool> edgemask(total, false);
        for( i = 4; i < total; i += 2 )
        {
            if( edgemask[i] )
                continue;
            Point2f a, b, c;
            int edge = i;
            int A = edgeOrg(edge, &a);
            if ( A < 4 ) continue;
            edgemask[edge] = true;
            edge = getEdge(edge, NEXT_AROUND_LEFT);
            int B = edgeOrg(edge, &b);
            if ( B < 4 ) continue;
            edgemask[edge] = true;
            edge = getEdge(edge, NEXT_AROUND_LEFT);
            int C = edgeOrg(edge, &c);
            if ( C < 4 ) continue;
            edgemask[edge] = true;

            ind.push_back(A);
            ind.push_back(B);
            ind.push_back(C);
        }
    }
};
