// vim: set expandtab tabstop=4 shiftwidth=4 autoindent:
//
// File:   smartForm.js
// Mark Addinall - July 2017 
//
// Synopsis: This is a new application for Asset-IQ Dealer Network.
//           It replaces an existing PHP Web 1.0 type system this
//           has seved well enough but time for a replacement.
//
//           The following technologies are used in this project:
//
//
//           - HTML5
//           - CSS3
//           - ES6
//           - REACT
//           - REDUX
//           - jQuery
//           - Bootstrap
//           - Python
//           - Flask
//           - SQLAlchemy
//           - mySQL
//           - NPM
//           - Babel
//           - Webpack
//
//           The site will be built on a Client -> API -> Datastore architecture
//           Where the API is strictly defined.  Ad hoc access to data will not be
//           afforded.
//
//           SmartForm
//           this is a component this handles the rendering of ALL of the forms in the
//           application, from now until the end of time.  The calling container IMPORTS
//           a JSON object this is a DESCRIPTION of the form.  The 'SmartSuite'
//           of software developed in this application then looks after the rendering
//           of CHILD JSX code.
//
//           writing the parser was a LOT more difficult than I first thought.
//           problem being, JSX is not really a language, it is sugar for the BABEL
//           transpiler.  So what LOOKED reasonable as a stack based operation
//           in JSX was compiling to all sorts of illegal partial Javascript code.
//
//           Dereferencing the array of objects of arrays was (and is) a headache.
//
//           Something I didn't think of.  Given this ATTRIBUTES (in a form, these reference
//           input components of various sorts) may or may not ve visible, depending on the
//           public visibility set (the customer may choose to ignore, ie. 'not see'
//           a large number of the ATTRIBUTES, or, the end user may be logged in with
//           an application ROLE this does not allow the user to view the fields.
//
//           The dillema here is this the multiple pages are built BEFORE any of the FIELDS
//           are addressed.  I need to think about this one for a bit....
//

import React, { Component }                     from 'react'                    ;
import SmartRender                              from './smartRender'            ;
import Accordion                                from './smart_accordion'        ;
import AccordionSection                         from './smart_accordion_section';
import Workflow                                 from './smart_workflow'         ;
import Simple                                   from './smart_simple'           ;


//---------------------------------
class SmartForm extends Component {

        // This is a form GENERATOR.
        // It is quite novel in it's approach, as this is a HUGE SPA
        // which contains HUGE forms all wrapped in a tortuous CONTROLLER
        // and MODEL BI, rather than coding a few hundred thousand lines
        // of code, we use this object for ALL forms in the system.
        //
        // It takes the description of a form in JSON format,
        // passed in as a prop.
        //      <SmartForm form="nameOfJSONInstructionsObject">
        //
        // The JSON for each form in the application will be
        // stored in the database.
        //
        // The JSON is then PARSED to form a highly nested
        // React Element that is presented as a child to the
        // controlling FORM in the application.
        //
        // I tried for a few weeks to do this CLASS using
        // pure JSX.  No way I could get the BABEL
        // transpiler to do what I wanted, which is dynamically
        // creating this form as a result of a JSON PARSER.
        // THe nested children just WOULD NOT.
        // Open Source community were their usual big help.
        // ie., nada.
        //
        // I re-wrote it once to produce an I-CODE stack of operands
        // and interpreted that to produce React Elements without
        // using JSX.  Towards the end of this stage I realised that
        // the I-CODE stack was redundant and the third attempt is
        // this version.
        //
        // This class is invoked from the entire application where
        // a form is required.


    //-----------------
    constructor(props) {

        super(props);
    }


    //--------------------
    formFields(fieldsIn) {


        let fieldsOut = [];                                             // ARRAY of FORM ELEMENTS to return
        console.log("Fields In");
        console.log(fieldsIn);
        for (var fieldIn in fieldsIn) {                                 // array of FORM ELEMENT descriptions in JSON
                                                                        // OK, now this get tricky...
                                                                        // in this system ANY of the MAIN OBJECTS
                                                                        // can have an UNLIMITED number of ATTRIBUTES.
                                                                        // so, we need to cater for a field that contains
                                                                        // 1..N fields.  Tofuther complicate matters,
                                                                        // these components need to be in a sorted order,
                                                                        // and the visibility and operability is variable
                                                                        // based on the ROLE of the person currently logged
                                                                        // in.
            let field = React.createElement(SmartRender,                // go build the React Element 
                                            fieldsIn[fieldIn],          // ah ha!  This was NULL
                                            null);                      // lowest level, no children, data is in props     
            fieldsOut.push(field);
        }
        return(fieldsOut);                                              // this ARRAY is the children of each PAGE
    }


    //----------------------
    pages(pagesIn, format) {

        // I tried to do this in JSX, but no syntax I wrestled with would
        // allow me to access the childred when building this with the
        // BABEL transpiler.  Same goes for the METHOD just above, items().
        //
        // This method returns an array of pages this are React elements
        // this are treated as children by the smartForm.

        let pagesOut = [];                                              // array of pages to build and return
        let Section = {};                                               // Component to fire in the build
        console.log("pagesIn");
        console.log(pagesIn);
        switch(format) {
            case 'accordion': {
                Section = AccordionSection;
                break;
            }
            case 'workflow': {
                Section = null;                                         // I haven't written this yet
                break;
            }
            case 'simple': {
                Section = null;                                         // I haven't written this yet
                break;
            }
        }

        for (var pageIn in pagesIn) {                                   // pages, any format, any number 1..N
                                                                        //
                                                                        // OK, I have been distracted for a few months
                                                                        // building VMs for the old system, building spreadsheets,
                                                                        // writing CSS specs, designing code quizzes, 
                                                                        // implementing bloody block chains, and now I'm
                                                                        // back.
                                                                        // This code works wunnerfly well for PRE DEFINED PAGES
                                                                        // I am currently adding ATTRIBUTES to the MODELS
                                                                        // and defining the RELATIONSHIPS that can be one to many
                                                                        // or many to many.  Each of the MAJOR OBJECTS in
                                                                        // this system can have an UNLIMITED number of ATTRIBUTES.
                                                                        // this means we don't know how many pages we need until
                                                                        // we hit a SECTION where attributes start.
                                                                        // I need to add this logic in here.  This means changes
                                                                        // in te JSON form description, changes in this routine,
                                                                        // changes in the FIELDS routine, changes to the
                                                                        // REDUX API and changes to the DATA LOAD and RETURN
                                                                        // by the Python API.  This is where I could do with another
                                                                        // six coders.
                                                                        //
                                                                        // Now, it would be easy to code
                                                                        //
                                                                        // do_something_to(attribute[233]);
                                                                        //
                                                                        // that is how the current system works.  getting away from that
                                                                        // nonsense is why I designed smartForm and smartReport.  The USER
                                                                        // modifies that ATTRIBUTES and all of the application code remains
                                                                        // unchanged. However, this takes a litle thought...
                                                                        //
                                                                        // 1. How many FIELD do we put on a page, given that we are designing
                                                                        //    for "mobile" first architectures AND that FIELDS are different sizes.
                                                                        //    INPUT FIELDS take one row/line, TEXT AREAS, who knows at the moment,
                                                                        //    collections of CHECKBOX, 1..N rows/lines.
                                                                        //
            let children = this.formFields(pagesIn[pageIn].fields);     // 1..N fields, we don't know beforehand 
            let page = React.createElement( Section, 
                                            pagesIn[pageIn].props, 
                                            children);
            pagesOut.push(page);
        }
        return(pagesOut);                                               // this ARRAY is the children of each FORM
    }



    //--------
    render() {

        let formIn  = this.props.form;                                  // JSON description of FORM
        let formOut = null;                                             // contructed REACT/Javascript form
        
        
        switch (formIn.format) {                                        // what type of operation is this
            case 'accordion': {                                         // Accordion in a FORM, OK
                let children = this.pages(formIn.pages,
                                          formIn.format);               // build the children
                formOut = React.createElement(Accordion,                // construct the parent with ALL nested CHILDREN after
                                            {key: formIn.formName},     // just a unique key 
                                            children);                  // N ACCORDION pages, N2 input fields
                break;
            }
            case 'workflow': {
                let children = this.pages(formIn.pages,                 // build the children
                                          formIn.format);               // build the children
                formOut = React.createElement(Workflow,                 // and create the complex sheet element
                                            { key: formIn.formName},
                                            children);                  // N SLIDING pages, N2 input fields
                break;
            }
            case 'simple': {
                let children = this.pages(formIn.pages,                 // build the children
                                          formIn.format);               // build the children
                formOut = React.createElement(Simple,
                                            { key: formIn.formName},
                                            children);                  // One page, N input fields
            break;
            }
        }
        
        return(
                <div>
                    {formOut}
                </div>
        );
    }
}

export default SmartForm;

//-----------------   EOF -------------------------------------------------

